import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import type { AgentId } from '../arena/types';
import type { PositionsArtifact } from '../experiment/assets';
import { createCanvasResizeObserver, createContextLossHandler, resizeRendererAndCamera, teardownWebglScene } from './lifecycle';
import { layoutPositions, partitionByRole, writeColors, type NeuronRole } from './activity-layout';
import { VIRIDIS_LUT } from './colormap';
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

interface ArmVisual {
  group: THREE.Group;
  roles: Record<NeuronRole, ArmRoleGroup>;
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
  private readonly arms: Record<AgentId, ArmVisual>;

  private reducedMotion: boolean;
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

      const layout = layoutPositions(options.positions.xyz, options.positions.positionSource);
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
    return { group, roles };
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
    if (this.disposed || this.contextLost) return;
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
    if (this.disposed || this.contextLost) return;
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
