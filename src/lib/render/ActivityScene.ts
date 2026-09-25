import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import type { AgentId } from '../arena/types';
import type { PositionsArtifact } from '../experiment/assets';
import { createCanvasResizeObserver, createContextLossHandler, resizeRendererAndCamera, teardownWebglScene } from './lifecycle';
import { layoutPositions, partitionByRole, writeColors, writeEffectColors, type NeuronRole } from './activity-layout';
import { DIVERGING_LUT, VIRIDIS_LUT } from './colormap';
import { POINT_SIZE } from './activity-constants';

/**
 * Read-only Three.js presentation layer for the anatomical activity view
 * (WP3). Shares `ArenaScene.ts`'s lifecycle/disposal/context-loss plumbing
 * via `./lifecycle.ts` (constructor rolls back a partially-built GPU context
 * on any throw; `dispose()` is idempotent and frees every GPU-owned resource
 * via `dispose.ts`), but is otherwise independent: it never touches
 * `src/lib/arena/*` simulation state, and its only per-tick input is each
 * arm's already-computed rate vector (`update(agentId, rates)`), never a
 * live reference into Worker/runner state.
 *
 * Draws both arms' 1,008 neurons at their MaleCNS soma positions (**Measured**),
 * colored by live computed rate (**Computed**), with role shown by point
 * shape (circle = sensory, square = bridge, triangle = descending —
 * **Annotated**), never by color alone. `activity-layout.ts` computes the
 * one shared (arm-agnostic) centered/scaled position set both arms are built
 * from — the two arms are offset horizontally as whole groups, never given
 * separately fabricated coordinates.
 */

export interface ActivitySceneOptions {
  canvas: HTMLCanvasElement;
  /** Element whose content-box size drives renderer/camera sizing. Defaults to `canvas.parentElement`. */
  container?: HTMLElement;
  /** The soma-position sidecar (`experiment/assets.ts#loadPositions`'s `status: 'ok'` payload). */
  positions: PositionsArtifact;
  /** Declared dynamics bounds (`ConnectomeGraph.metadata.rateMin`/`rateMax`) the colormap is scaled against — never a per-frame auto-normalized range. */
  rateMin: number;
  rateMax: number;
  /** Mirrors `prefers-reduced-motion`: disables camera damping. Color-update cadence throttling is the host's responsibility (see `ActivityPanel.svelte`). */
  reducedMotion?: boolean;
  /** Fired when the WebGL context is lost after a successful start. The host is responsible for disposing this instance. */
  onContextLost?: (info: { reason: string }) => void;
}

/** Thrown synchronously from the constructor when a WebGL context cannot be created. Distinct from `ArenaSceneUnavailableError` — this scene names itself, not the arena. */
export class ActivitySceneUnavailableError extends Error {
  constructor(cause?: unknown) {
    super('Activity view: could not create a WebGL context');
    this.name = 'ActivitySceneUnavailableError';
    if (cause !== undefined) this.cause = cause;
  }
}

/** Horizontal offset of each arm's group from the shared origin, in the same normalized units `layoutPositions` scales soma positions into (main cloud extent `[-1, 1]`). Large enough that neither arm's point cloud (nor its unavailable strip) ever overlaps the other's. */
const ARM_OFFSET = 1.7;
const ROLES: readonly NeuronRole[] = ['sensory', 'bridge', 'descending'];
/**
 * Neutral "no computed rate yet" color: a mid grey, deliberately outside
 * `VIRIDIS_LUT`'s hue range (which runs dark purple to yellow) so it can
 * never be mistaken for a real low/high rate reading. Used before the first
 * `update()` for an arm (points would otherwise default to `(0, 0, 0)` —
 * black, indistinguishable from the scene's near-black background, i.e. the
 * view would look empty rather than "not yet streaming") and again whenever
 * `clear()` is called (e.g. after `ExperimentRunner#reset()` clears
 * `latestRates` — see `ActivityPanel.svelte`'s `frame()`), so a stale
 * previous run's colors are never left on screen under a "Computed rate"
 * label that no longer describes them.
 */
const NO_DATA_COLOR: readonly [number, number, number] = [0.32, 0.35, 0.4];
type PointShape = 'circle' | 'square' | 'triangle';
const ROLE_SHAPE: Record<NeuronRole, PointShape> = {
  sensory: 'circle',
  bridge: 'square',
  descending: 'triangle'
};

/**
 * Lesion-effect mode's FDR-significance marker (WP3): a hollow ring drawn at
 * every neuron whose effect did NOT survive Benjamini-Hochberg FDR
 * correction (`emphasize[i] === false` — see `activity-layout.ts#writeEffectColors`'s
 * own doc comment). Shape, not hue — a non-color-only signal alongside (not
 * instead of) that function's color-blend-toward-neutral, so significance is
 * never encoded by color/saturation alone. Slightly larger than the base
 * point sprite so it reads as an outline/halo around the underlying role
 * shape rather than occluding it.
 */
const OUTLINE_RING_COLOR = '#ffe08a';
const OUTLINE_RING_SIZE_FACTOR = 1.7;

/** Draw a hollow ring into a small offscreen canvas, once — the lesion-effect mode's "not FDR-significant" marker texture (see `OUTLINE_RING_COLOR`'s doc comment). */
const buildOutlineRingTexture = (): THREE.CanvasTexture => {
  const canvas = document.createElement('canvas');
  canvas.width = 32;
  canvas.height = 32;
  const context = canvas.getContext('2d');
  if (context) {
    context.strokeStyle = '#ffffff';
    context.lineWidth = 3;
    context.beginPath();
    context.arc(16, 16, 12, 0, Math.PI * 2);
    context.stroke();
  }
  return new THREE.CanvasTexture(canvas);
};

/** Draw one filled shape into a small offscreen canvas, once, for use as a `THREE.PointsMaterial#map`. No per-frame allocation — built only at construction. */
const buildShapeTexture = (shape: PointShape): THREE.CanvasTexture => {
  const canvas = document.createElement('canvas');
  canvas.width = 32;
  canvas.height = 32;
  const context = canvas.getContext('2d');
  if (context) {
    context.fillStyle = '#ffffff';
    context.beginPath();
    if (shape === 'circle') {
      context.arc(16, 16, 14, 0, Math.PI * 2);
    } else if (shape === 'square') {
      context.rect(3, 3, 26, 26);
    } else {
      context.moveTo(16, 2);
      context.lineTo(30, 29);
      context.lineTo(2, 29);
      context.closePath();
    }
    context.fill();
  }
  const texture = new THREE.CanvasTexture(canvas);
  return texture;
};

interface ArmRoleGroup {
  points: THREE.Points;
  colorAttribute: THREE.BufferAttribute;
  /** Neuron indices (into a `rates` array) this role group's points were built from, in the same order as its position/color attributes. */
  indices: Int32Array;
}

/** Lesion-effect mode's per-arm "not FDR-significant" ring overlay (see `OUTLINE_RING_COLOR`'s doc comment) — one `THREE.Points` per arm, spanning every role, rebuilt (not preallocated) each time `setStaticColors`/`setNoLesionData` runs, since that only happens on mode entry or a topology switch while in lesion mode, never per animation frame. */
interface ArmOutline {
  points: THREE.Points;
}

interface ArmVisual {
  group: THREE.Group;
  roles: Record<NeuronRole, ArmRoleGroup>;
  outline: ArmOutline;
}

export class ActivityScene {
  private readonly canvas: HTMLCanvasElement;
  private readonly container: HTMLElement;
  private readonly scene = new THREE.Scene();
  private readonly camera: THREE.PerspectiveCamera;
  private readonly renderer: THREE.WebGLRenderer;
  private readonly controls: OrbitControls;
  private readonly onContextLost: ((info: { reason: string }) => void) | undefined;
  private readonly rateMin: number;
  private readonly rateMax: number;
  private readonly shapeTextures: Record<PointShape, THREE.CanvasTexture>;
  private readonly materials: Record<NeuronRole, THREE.PointsMaterial>;
  private readonly outlineTexture: THREE.CanvasTexture;
  private readonly outlineMaterial: THREE.PointsMaterial;
  private readonly arms: Record<AgentId, ArmVisual>;
  /** The shared, arm-agnostic centered/scaled position array `layoutPositions` produced (`ActivitySceneOptions.positions` in, once, at construction) — kept so `setStaticColors`/`setNoLesionData` can (re)build each arm's outline-ring overlay (`ArmOutline`) from arbitrary neuron-index subsets without re-running `layoutPositions`. Both arms share this same array (their `group.position.x` offsets, not separate coordinates, are what make the arms visually distinct — see `buildArm`). */
  private readonly basePositions: Float32Array;

  private reducedMotion: boolean;
  /** `'live'` (the default): `update()`/`clear()` write per-tick colors as usual. `'lesion'`: both become no-ops so `setStaticColors`'s static colors persist untouched, and both arms' outline overlays are hidden on entry — see `setMode`'s own doc comment. */
  private mode: 'live' | 'lesion' = 'live';
  private resizeObserver: ResizeObserver | undefined;
  private contextLost = false;
  private disposed = false;

  private readonly handleContextLost = createContextLossHandler(() => {
    this.contextLost = true;
    this.onContextLost?.({ reason: 'webglcontextlost' });
  });

  constructor(options: ActivitySceneOptions) {
    this.canvas = options.canvas;
    this.container = options.container ?? options.canvas.parentElement ?? options.canvas;
    this.reducedMotion = options.reducedMotion ?? false;
    this.onContextLost = options.onContextLost;
    this.rateMin = options.rateMin;
    this.rateMax = options.rateMax;

    try {
      this.renderer = new THREE.WebGLRenderer({ canvas: this.canvas, antialias: true, alpha: false });
    } catch (error) {
      throw new ActivitySceneUnavailableError(error);
    }

    // Same rollback discipline as `ArenaScene`'s constructor: once the
    // WebGLRenderer exists it owns a real GPU context, so anything past this
    // point that throws must free it before propagating.
    let controls: OrbitControls | undefined;
    try {
      this.renderer.setPixelRatio(Math.min(window.devicePixelRatio ?? 1, 2));
      this.renderer.outputColorSpace = THREE.SRGBColorSpace;
      this.canvas.addEventListener('webglcontextlost', this.handleContextLost, false);

      this.scene.background = new THREE.Color('#05070c');

      this.camera = new THREE.PerspectiveCamera(45, 1, 0.05, 40);
      this.camera.position.set(0, 1.7, 4.6);

      controls = new OrbitControls(this.camera, this.renderer.domElement);
      this.controls = controls;
      this.controls.enableDamping = !this.reducedMotion;
      this.controls.dampingFactor = 0.08;
      this.controls.target.set(0, -0.2, 0);
      this.controls.update();

      this.shapeTextures = {
        circle: buildShapeTexture('circle'),
        square: buildShapeTexture('square'),
        triangle: buildShapeTexture('triangle')
      };
      this.materials = {
        sensory: this.buildMaterial('sensory'),
        bridge: this.buildMaterial('bridge'),
        descending: this.buildMaterial('descending')
      };
      this.outlineTexture = buildOutlineRingTexture();
      this.outlineMaterial = new THREE.PointsMaterial({
        size: POINT_SIZE * OUTLINE_RING_SIZE_FACTOR,
        map: this.outlineTexture,
        color: new THREE.Color(OUTLINE_RING_COLOR),
        transparent: true,
        alphaTest: 0.2,
        depthWrite: false,
        sizeAttenuation: true
      });

      const layout = layoutPositions(options.positions.xyz, options.positions.positionSource);
      this.basePositions = layout.points;
      const partition = partitionByRole(options.positions.role);
      const roleIndices: Record<NeuronRole, Int32Array> = {
        sensory: partition.sensoryIdx,
        bridge: partition.bridgeIdx,
        descending: partition.descendingIdx
      };

      this.arms = {
        left: this.buildArm(layout.points, roleIndices, -ARM_OFFSET),
        right: this.buildArm(layout.points, roleIndices, ARM_OFFSET)
      };

      this.setupResizeObserver();
      const initialWidth = this.container.clientWidth || 1;
      const initialHeight = this.container.clientHeight || 1;
      this.resize(initialWidth, initialHeight);
    } catch (error) {
      teardownWebglScene({
        canvas: this.canvas,
        handleContextLost: this.handleContextLost,
        resizeObserver: this.resizeObserver,
        scene: this.scene,
        renderer: this.renderer,
        controls
      });
      throw error;
    }
  }

  private buildMaterial(role: NeuronRole): THREE.PointsMaterial {
    return new THREE.PointsMaterial({
      size: POINT_SIZE,
      map: this.shapeTextures[ROLE_SHAPE[role]],
      vertexColors: true,
      transparent: true,
      alphaTest: 0.4,
      depthWrite: false,
      sizeAttenuation: true
    });
  }

  private buildArm(
    basePositions: Float32Array,
    roleIndices: Record<NeuronRole, Int32Array>,
    offsetX: number
  ): ArmVisual {
    const group = new THREE.Group();
    group.position.x = offsetX;
    this.scene.add(group);

    const roles = {} as Record<NeuronRole, ArmRoleGroup>;
    for (const role of ROLES) {
      const indices = roleIndices[role];
      const positionArray = new Float32Array(indices.length * 3);
      for (let slot = 0; slot < indices.length; slot += 1) {
        const source = indices[slot] * 3;
        const destination = slot * 3;
        positionArray[destination] = basePositions[source];
        positionArray[destination + 1] = basePositions[source + 1];
        positionArray[destination + 2] = basePositions[source + 2];
      }
      const colorArray = new Float32Array(indices.length * 3);
      // Start every point at the neutral "no data" color rather than the
      // typed array's zero-fill default (0, 0, 0 — black, invisible against
      // the near-black background) — see `NO_DATA_COLOR`'s doc comment.
      for (let component = 0; component < colorArray.length; component += 3) {
        colorArray[component] = NO_DATA_COLOR[0];
        colorArray[component + 1] = NO_DATA_COLOR[1];
        colorArray[component + 2] = NO_DATA_COLOR[2];
      }
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute('position', new THREE.BufferAttribute(positionArray, 3));
      const colorAttribute = new THREE.BufferAttribute(colorArray, 3);
      geometry.setAttribute('color', colorAttribute);
      const points = new THREE.Points(geometry, this.materials[role]);
      group.add(points);
      roles[role] = { points, colorAttribute, indices };
    }

    // Lesion-effect mode's "not FDR-significant" ring overlay (see
    // `OUTLINE_RING_COLOR`'s doc comment) — starts empty/invisible; only
    // `setStaticColors`/`setNoLesionData` ever populate or show it.
    const outlineGeometry = new THREE.BufferGeometry();
    outlineGeometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(0), 3));
    const outlinePoints = new THREE.Points(outlineGeometry, this.outlineMaterial);
    outlinePoints.visible = false;
    group.add(outlinePoints);

    return { group, roles, outline: { points: outlinePoints } };
  }

  /** Mirror a live `prefers-reduced-motion` change: toggles camera damping immediately. */
  setReducedMotion(value: boolean): void {
    if (this.disposed || this.reducedMotion === value) return;
    this.reducedMotion = value;
    this.controls.enableDamping = !value;
  }

  /**
   * Write `agentId`'s colors from this tick's full per-neuron `rates`
   * (length `neuronCount`, indexed the same way as
   * `experiment/assets.ts#PositionsArtifact`'s arrays) into that arm's three
   * preallocated role color attributes and mark them for upload. No
   * allocation: `writeColors` (`activity-layout.ts`) writes directly into
   * each `THREE.BufferAttribute`'s backing array. Never called for the arm
   * that is not currently streaming — the host (`ActivityPanel.svelte`)
   * only calls this when `ExperimentRunner#getLatestRates` actually returns
   * a fresh array for that arm.
   */
  update(agentId: AgentId, rates: Float32Array): void {
    // No-op while in lesion mode (WP3): the lesion-effect color mode's
    // static colors (`setStaticColors` below) must persist untouched across
    // frames — `ActivityPanel.svelte`'s `frame()` already skips its whole
    // rates-polling block while `mode === 'lesion'` (so this branch is
    // normally never reached then), but gating here too is defense in depth
    // against any other future caller.
    if (this.disposed || this.contextLost || this.mode === 'lesion') return;
    const arm = this.arms[agentId];
    for (const role of ROLES) {
      const group = arm.roles[role];
      writeColors(rates, group.indices, this.rateMin, this.rateMax, VIRIDIS_LUT, group.colorAttribute.array as Float32Array);
      group.colorAttribute.needsUpdate = true;
    }
  }

  /**
   * Repaint `agentId`'s points back to the neutral "no data" color (see
   * `NO_DATA_COLOR`) — the counterpart to `update()`, for when this arm no
   * longer has fresh rates to show (e.g. `ExperimentRunner#reset()` cleared
   * `latestRates`; the host calls this instead of leaving the previous
   * run's final colors on screen — see `ActivityPanel.svelte`'s `frame()`).
   */
  clear(agentId: AgentId): void {
    // No-op while in lesion mode — same reasoning as `update()` above: this
    // repaint-to-neutral is a *live*-mode concept ("no fresh rate this
    // frame"), and must never overwrite the lesion-effect mode's static
    // colors. `setNoLesionData` below is the lesion-mode counterpart for an
    // arm with no atlas coverage (e.g. disconnected), and paints the same
    // neutral color but is never gated on `mode`.
    if (this.disposed || this.contextLost || this.mode === 'lesion') return;
    this.paintNeutral(agentId);
  }

  /** Shared neutral-repaint body for `clear()` (live mode only) and `setNoLesionData()` (lesion mode, always) — see each method's own doc comment. */
  private paintNeutral(agentId: AgentId): void {
    const arm = this.arms[agentId];
    for (const role of ROLES) {
      const array = arm.roles[role].colorAttribute.array as Float32Array;
      for (let component = 0; component < array.length; component += 3) {
        array[component] = NO_DATA_COLOR[0];
        array[component + 1] = NO_DATA_COLOR[1];
        array[component + 2] = NO_DATA_COLOR[2];
      }
      arm.roles[role].colorAttribute.needsUpdate = true;
    }
  }

  /**
   * Lesion-effect mode: paint `agentId`'s points from a static per-neuron
   * `effect`/`emphasize` (FDR-significant) pair — the shipped lesion atlas's
   * data for whichever graph key `agentId`'s current topology maps to (see
   * `experiment/lesionAtlas.ts#lesionAtlasGraphKeyForTopology`). Always
   * writes (not gated on `mode`, unlike `update()`/`clear()`): the host
   * (`ActivityPanel.svelte`) only calls this while `mode === 'lesion'`, and
   * gating here too would just make a caller bug silently do nothing instead
   * of writing wrong-looking colors, which is worse to debug. Also (re)builds
   * this arm's "not FDR-significant" outline-ring overlay (see
   * `OUTLINE_RING_COLOR`'s doc comment) from `emphasize`.
   */
  setStaticColors(agentId: AgentId, effect: ArrayLike<number>, emphasize: ArrayLike<boolean>, absMax: number): void {
    if (this.disposed || this.contextLost) return;
    const arm = this.arms[agentId];
    for (const role of ROLES) {
      const group = arm.roles[role];
      writeEffectColors(effect, emphasize, group.indices, absMax, DIVERGING_LUT, group.colorAttribute.array as Float32Array);
      group.colorAttribute.needsUpdate = true;
    }
    this.paintOutline(agentId, emphasize);
  }

  /**
   * Lesion-effect mode's honest "no data" state for an arm whose current
   * topology the atlas does not cover (disconnected — see
   * `lesionAtlasGraphKeyForTopology`'s doc comment). Paints the same neutral
   * grey `clear()` uses (never a fabricated effect color) and hides that
   * arm's outline overlay (there is no per-neuron significance to show).
   */
  setNoLesionData(agentId: AgentId): void {
    if (this.disposed || this.contextLost) return;
    this.paintNeutral(agentId);
    const arm = this.arms[agentId];
    arm.outline.points.visible = false;
  }

  /**
   * (Re)build `agentId`'s outline-ring overlay from `emphasize` — one point
   * per neuron whose effect did *not* survive FDR correction, in the same
   * shared/arm-agnostic centered coordinate space `this.basePositions`
   * already holds (the overlay is a child of this arm's own `group`, which
   * carries the arm's `offsetX` transform — see `buildArm` — so no manual
   * offset is needed here). A fresh `Float32Array`/`BufferAttribute` each
   * call rather than a preallocated, `setDrawRange`-trimmed buffer: this
   * only ever runs on lesion-mode entry or a topology switch while lesion
   * mode is active, never per animation frame, so the allocation is outside
   * this scene's actual "no allocation" hot path (`update()`/`render()`).
   */
  private paintOutline(agentId: AgentId, emphasize: ArrayLike<boolean>): void {
    const arm = this.arms[agentId];
    const neuronCount = this.basePositions.length / 3;
    let count = 0;
    for (let neuron = 0; neuron < neuronCount; neuron += 1) if (!emphasize[neuron]) count += 1;
    const positions = new Float32Array(count * 3);
    let writeIndex = 0;
    for (let neuron = 0; neuron < neuronCount; neuron += 1) {
      if (emphasize[neuron]) continue;
      const source = neuron * 3;
      const destination = writeIndex * 3;
      positions[destination] = this.basePositions[source];
      positions[destination + 1] = this.basePositions[source + 1];
      positions[destination + 2] = this.basePositions[source + 2];
      writeIndex += 1;
    }
    // Round-2 dual review (Suggestion, latent-bug class): `setAttribute`
    // alone only reassigns `geometry.attributes.position` — it neither frees
    // the *previous* attribute's underlying GL buffer (three's
    // `WebGLAttributes` only releases it on the geometry's own `dispose()`)
    // nor recomputes `geometry.boundingSphere` (computed once, lazily, on
    // first visible render, then cached) — so a later topology switch in
    // lesion mode, which changes this arm's non-significant subset size,
    // would otherwise leave a stale culling sphere and an orphaned GPU
    // buffer waiting on garbage collection. `geometry.dispose()` here frees
    // only the geometry's own GPU-owned buffers (not the shared
    // `outlineMaterial`/`outlineTexture`, which are never touched by a
    // geometry's own `dispose()`), so this is safe to call on every paint.
    const geometry = arm.outline.points.geometry;
    geometry.dispose();
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.computeBoundingSphere();
    arm.outline.points.visible = count > 0;
  }

  /**
   * Switch between `'live'` (default: `update()`/`clear()` write per-tick
   * colors as usual) and `'lesion'` (both become no-ops, so whatever
   * `setStaticColors`/`setNoLesionData` last painted persists untouched
   * across frames — see those methods and `update()`/`clear()`'s own doc
   * comments). Switching back to `'live'` also hides both arms' outline
   * overlays (a lesion-mode-only marker) so a subsequent Live-mode frame
   * never shows a stale "not FDR-significant" ring over live rate colors.
   */
  setMode(mode: 'live' | 'lesion'): void {
    if (this.disposed || this.mode === mode) return;
    this.mode = mode;
    if (mode === 'live') {
      for (const agentId of Object.keys(this.arms) as AgentId[]) {
        this.arms[agentId].outline.points.visible = false;
      }
    }
  }

  /** Explicit resize hook, also used internally by the `ResizeObserver` callback. */
  resize(width: number, height: number): void {
    if (this.disposed) return;
    resizeRendererAndCamera(this.renderer, this.camera, width, height);
  }

  private setupResizeObserver(): void {
    this.resizeObserver = createCanvasResizeObserver(this.container, (width, height) => this.resize(width, height));
  }

  /**
   * Render one frame. Cheap and allocation-free — camera/control update plus
   * one `renderer.render()` call; all per-tick work happens in `update()`.
   * Called by the host once per animation frame while the panel is open,
   * independent of whether a fresh `update()` happened this frame (so the
   * camera keeps responding to drag input even between rate updates).
   *
   * `nowMs` mirrors `ArenaScene#update`'s signature (the host's own rAF
   * timestamp) for API parity; this scene has no frame-timing telemetry of
   * its own to derive from it today, so it is currently unused.
   */
  render(nowMs: number = performance.now()): void {
    void nowMs;
    if (this.disposed || this.contextLost) return;
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }

  /** Stop observing/listening and free every GPU-owned resource. Idempotent. */
  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    teardownWebglScene({
      canvas: this.canvas,
      handleContextLost: this.handleContextLost,
      resizeObserver: this.resizeObserver,
      scene: this.scene,
      renderer: this.renderer,
      controls: this.controls,
      skipForceContextLoss: this.contextLost,
      clearScene: true
    });
  }
}
