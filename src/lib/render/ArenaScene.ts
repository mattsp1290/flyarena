import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { ARENA_CONFIG, type ArenaConfig } from '../arena/config';
import type { AgentId, ArenaSnapshot, FoodState, Vec2 } from '../arena/types';
import type { GraphMode } from '../connectome/format';
import { createCanvasResizeObserver, createContextLossHandler, resizeRendererAndCamera, teardownWebglScene } from './lifecycle';
import {
  agentTransform,
  detectFoodPickups,
  detectHazardContacts,
  pushTrailPoint,
  type FoodRespawnRecord,
  type HazardContactResult
} from './transforms';

/**
 * Read-only Three.js presentation layer for the arena.
 *
 * `ArenaScene` never touches `src/lib/arena/*` simulation state beyond
 * reading the `ArenaSnapshot` it is handed each frame through `update()`.
 * It holds no reference back into the world model, has no sensing/decision
 * logic, and camera/control state is purely local to this class — nothing
 * here can influence observations or the physics step. See
 * `docs/architecture.md`'s "Renderer" boundary and the closed-loop
 * contract for why that separation matters.
 */

export interface FrameTelemetry {
  /** Exponential-moving-average frames per second, from renderer frame timing only — never neural/simulated time. */
  fps: number;
  /** Milliseconds elapsed since the previous `update()` call. */
  frameMs: number;
}

/** The subset of `ArenaConfig` the renderer reads: bounds and entity radii. */
export type RenderArenaConfig = Readonly<
  Pick<ArenaConfig, 'halfWidth' | 'halfDepth' | 'agentRadius' | 'foodRadius' | 'hazardRadius'>
>;

export interface ArenaSceneOptions {
  canvas: HTMLCanvasElement;
  /** Element whose content-box size drives renderer/camera sizing. Defaults to `canvas.parentElement`. */
  container?: HTMLElement;
  /** Defaults to the shared `ARENA_CONFIG`; only bounds/radii fields are read. Pass the world's own retained config if it differs from the default. */
  arenaConfig?: RenderArenaConfig;
  /** Mirrors `prefers-reduced-motion`: disables camera damping, hazard spin, agent trails, and effect growth animation. */
  reducedMotion?: boolean;
  onFrame?: (telemetry: FrameTelemetry) => void;
  /** Fired when the WebGL context is lost after a successful start. The host is responsible for disposing this instance and constructing a new one to recover. */
  onContextLost?: (info: { reason: string }) => void;
}

/** Thrown synchronously from the constructor when a WebGL context cannot be created. */
export class ArenaSceneUnavailableError extends Error {
  constructor(cause: unknown) {
    super(
      `ArenaScene: could not create a WebGL context (${cause instanceof Error ? cause.message : String(cause)})`
    );
    this.name = 'ArenaSceneUnavailableError';
  }
}

/**
 * Fixed per-*slot* (left/right position), not per-topology, accent used for
 * the body material and trail color. This never changes when an arm's
 * topology switches — see `buildAgentBody`'s doc comment for why body
 * shape/color identify *which agent*, never *which topology* is currently
 * running on it.
 */
const AGENT_SLOT_ACCENT: Record<AgentId, string> = {
  left: '#f4c95d',
  right: '#7fd7ff'
};

/**
 * Text + accent for each of the three labels a slot's label sprite can
 * show. All three are preallocated per agent at construction time (see
 * `buildAgentBody`) and toggled via `visible` — never rebuilt — so
 * `setAgentTopology` never allocates a canvas/texture/material per call.
 * `disconnected` gets its own warning-toned accent (distinct from either
 * slot's body accent) so a negative-control arm reads as visually distinct
 * from both "biological" and "rewired", not merely as a third color choice.
 */
const TOPOLOGY_LABEL: Record<GraphMode, { text: string; accent: string }> = {
  biological: { text: 'BIO', accent: '#f4c95d' },
  rewired: { text: 'REWIRED', accent: '#7fd7ff' },
  disconnected: { text: 'DISCONNECTED', accent: '#ef476f' }
};

/**
 * Derived from `TOPOLOGY_LABEL`'s own keys (bb45 follow-up) rather than a
 * separately hand-maintained literal array: `TOPOLOGY_LABEL` is a
 * `Record<GraphMode, ...>`, so it is already a complete, exhaustive map over
 * every `GraphMode` — a hand-written second list next to it could silently
 * drift out of sync with a future `GraphMode` addition (TypeScript would
 * still catch a *missing* key in `TOPOLOGY_LABEL` itself via its `Record`
 * type, but nothing previously forced this array to stay in step with it).
 * `Object.keys` returns `string[]`, so the cast is required; it is sound
 * specifically because `TOPOLOGY_LABEL` is typed as `Record<GraphMode, ...>`
 * with no index signature, meaning its keys can only ever be `GraphMode`.
 */
const TOPOLOGY_MODES = Object.keys(TOPOLOGY_LABEL) as readonly GraphMode[];

/**
 * Pure `mode -> label` mapping, exported for direct unit testing: like
 * `tracePanelPath` below, `ArenaScene` itself cannot be constructed under
 * jsdom (no WebGL), so the honesty-critical mapping this scene renders from
 * is tested here directly rather than only indirectly through a mounted
 * scene. Regression coverage for a real shipped bug: a fixed "BIO" label
 * that never tracked the arm's actual topology (see `setAgentTopology`'s
 * doc comment).
 */
export const topologyLabelFor = (mode: GraphMode): { text: string; accent: string } => TOPOLOGY_LABEL[mode];

const FPS_SMOOTHING = 0.15;
const TRAIL_CAPACITY = 240;
const TRAIL_LIFT = 0.05;
const EFFECT_POOL_SIZE = 16;
const EFFECT_LIFE_SECONDS = 0.6;
const EFFECT_BASE_SCALE_MULTIPLIER = 1.4;
const EFFECT_MAX_GROWTH = 1.6;
const FOOD_EFFECT_COLOR = new THREE.Color('#f4c95d');
const HAZARD_EFFECT_COLOR = new THREE.Color('#ef476f');
/** Clamp on the per-`update()` frame delta used for animation/FPS telemetry, so a backgrounded/throttled tab cannot report a nonsensical FPS spike or drive a huge single-step hazard spin. This is unrelated to `App.svelte`'s own simulation catch-up cap. */
const MAX_FRAME_DELTA_MS = 250;
const HAZARD_SPIN_RADIANS_PER_SECOND = 1.2;
const CONTROLS_DAMPING_FACTOR = 0.08;
const WALL_HEIGHT = 0.6;
const WALL_THICKNESS = 0.15;
const LABEL_SPRITE_SCALE: readonly [number, number] = [1.6, 0.6];
const LABEL_HEIGHT_FACTOR = 2.4;
const TWO_PI = Math.PI * 2;

/**
 * Draw a rounded rect if the browser supports it (Firefox < 112 / Safari <
 * 16 do not), else a plain rect. Exported so the fallback branch — which
 * `ArenaScene`'s own constructor can only exercise on a real WebGL context,
 * i.e. not under jsdom — is directly unit-testable.
 */
export const tracePanelPath = (context: CanvasRenderingContext2D, width: number, height: number): void => {
  context.beginPath();
  if (typeof context.roundRect === 'function') {
    context.roundRect(4, 20, width - 8, height - 40, 16);
  } else {
    context.rect(4, 20, width - 8, height - 40);
  }
};

/** Build a billboard label sprite once, from an offscreen canvas. No per-frame allocation. */
const createLabelSprite = (text: string, accent: string): THREE.Sprite => {
  const canvas = document.createElement('canvas');
  canvas.width = 256;
  canvas.height = 96;
  const context = canvas.getContext('2d');
  if (context) {
    context.fillStyle = 'rgba(7, 11, 19, 0.82)';
    tracePanelPath(context, canvas.width, canvas.height);
    context.fill();
    context.strokeStyle = accent;
    context.lineWidth = 3;
    context.stroke();
    context.fillStyle = accent;
    context.font = 'bold 40px system-ui, sans-serif';
    context.textAlign = 'center';
    context.textBaseline = 'middle';
    context.fillText(text, canvas.width / 2, canvas.height / 2 + 2);
  }
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  const material = new THREE.SpriteMaterial({ map: texture, transparent: true, depthWrite: false });
  const sprite = new THREE.Sprite(material);
  sprite.scale.set(LABEL_SPRITE_SCALE[0], LABEL_SPRITE_SCALE[1], 1);
  return sprite;
};

interface AgentVisual {
  group: THREE.Group;
  /** All three preallocated per-topology label sprites for this agent — see `buildAgentBody`. Exactly one is `visible` at a time; `setAgentTopology` only ever toggles `visible`, never creates or replaces one. */
  labelSprites: Record<GraphMode, THREE.Sprite>;
  trail: {
    positions: Float32Array;
    filled: number;
    geometry: THREE.BufferGeometry;
    line: THREE.Line;
  };
}

interface PooledEffect {
  sprite: THREE.Sprite;
  material: THREE.SpriteMaterial;
  life: number;
}

/**
 * Build one agent's body with unmistakable, non-color-only identity by
 * *slot*: the left slot is a smooth low-poly icosahedron; the right slot is
 * an angular octahedron with a contrasting wireframe overlay (evoking
 * rewired circuitry). This shape/color pairing is fixed by slot for the
 * lifetime of the scene — it identifies *which agent* (left/right), never
 * which topology is currently running on it. Both bodies also carry a nose
 * cone so heading is legible from any camera angle.
 *
 * The label sprite above each body is the only thing that identifies
 * topology, and it is the only part of this group `setAgentTopology` ever
 * touches — see that method's doc comment for why the two must never be
 * conflated. All three possible labels (`TOPOLOGY_LABEL`) are built once,
 * right here, and added to the group with only one `visible` at a time, so
 * a later topology switch never allocates a canvas/texture/material.
 */
const buildAgentBody = (
  id: AgentId,
  radius: number
): { group: THREE.Group; labelSprites: Record<GraphMode, THREE.Sprite> } => {
  const group = new THREE.Group();
  const accent = AGENT_SLOT_ACCENT[id];

  if (id === 'left') {
    const geometry = new THREE.IcosahedronGeometry(radius, 1);
    const material = new THREE.MeshStandardMaterial({
      color: accent,
      flatShading: true,
      roughness: 0.55,
      metalness: 0.05
    });
    group.add(new THREE.Mesh(geometry, material));
  } else {
    const geometry = new THREE.OctahedronGeometry(radius, 0);
    const material = new THREE.MeshStandardMaterial({
      color: accent,
      flatShading: true,
      roughness: 0.3,
      metalness: 0.25
    });
    const body = new THREE.Mesh(geometry, material);
    group.add(body);
    const edges = new THREE.LineSegments(
      new THREE.EdgesGeometry(geometry),
      new THREE.LineBasicMaterial({ color: '#0b3a4a' })
    );
    group.add(edges);
  }

  const nose = new THREE.Mesh(
    new THREE.ConeGeometry(radius * 0.35, radius * 0.9, 8),
    new THREE.MeshStandardMaterial({ color: '#f5f7fb', flatShading: true })
  );
  nose.rotation.x = Math.PI / 2;
  nose.position.z = radius * 0.95;
  group.add(nose);

  const labelSprites = {} as Record<GraphMode, THREE.Sprite>;
  for (const mode of TOPOLOGY_MODES) {
    const label = TOPOLOGY_LABEL[mode];
    const sprite = createLabelSprite(label.text, label.accent);
    sprite.position.y = radius * LABEL_HEIGHT_FACTOR;
    sprite.visible = false;
    group.add(sprite);
    labelSprites[mode] = sprite;
  }
  // Default visible label mirrors this slot's conventional default topology
  // (`App.svelte`'s initial `topology` state), so the scene never shows no
  // label at all before the host's first explicit `setAgentTopology` call —
  // the host still calls `setAgentTopology` once right after construction
  // to reconcile this against whatever the *actual* current topology is by
  // the time the scene exists (see that method's doc comment).
  const defaultMode: GraphMode = id === 'left' ? 'biological' : 'rewired';
  labelSprites[defaultMode].visible = true;

  return { group, labelSprites };
};

const buildTrail = (color: string): AgentVisual['trail'] => {
  const positions = new Float32Array(TRAIL_CAPACITY * 3);
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geometry.setDrawRange(TRAIL_CAPACITY, 0);
  const material = new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.55 });
  const line = new THREE.Line(geometry, material);
  line.frustumCulled = false;
  return { positions, filled: 0, geometry, line };
};

const buildEffectPool = (): PooledEffect[] => {
  const canvas = document.createElement('canvas');
  canvas.width = 64;
  canvas.height = 64;
  const context = canvas.getContext('2d');
  if (context) {
    const gradient = context.createRadialGradient(32, 32, 0, 32, 32, 32);
    gradient.addColorStop(0, 'rgba(255,255,255,0.95)');
    gradient.addColorStop(1, 'rgba(255,255,255,0)');
    context.fillStyle = gradient;
    context.fillRect(0, 0, 64, 64);
  }
  const texture = new THREE.CanvasTexture(canvas);
  const pool: PooledEffect[] = [];
  for (let index = 0; index < EFFECT_POOL_SIZE; index += 1) {
    const material = new THREE.SpriteMaterial({
      map: texture,
      transparent: true,
      opacity: 0,
      depthWrite: false
    });
    const sprite = new THREE.Sprite(material);
    sprite.visible = false;
    pool.push({ sprite, material, life: 0 });
  }
  return pool;
};

export class ArenaScene {
  private readonly options: ArenaSceneOptions;
  private readonly config: RenderArenaConfig;
  private reducedMotion: boolean;

  private readonly scene = new THREE.Scene();
  private readonly camera: THREE.PerspectiveCamera;
  private readonly renderer: THREE.WebGLRenderer;
  private readonly controls: OrbitControls;
  private readonly canvas: HTMLCanvasElement;
  private readonly container: HTMLElement;
  private resizeObserver: ResizeObserver | undefined;

  private readonly agentVisuals = new Map<AgentId, AgentVisual>();
  private readonly foodGeometry: THREE.SphereGeometry;
  private readonly foodMaterial: THREE.MeshStandardMaterial;
  private readonly foodPool: THREE.Mesh[] = [];
  private readonly foodGroup = new THREE.Group();
  private readonly hazardGeometry: THREE.IcosahedronGeometry;
  private readonly hazardMaterial: THREE.MeshStandardMaterial;
  private readonly hazardPool: THREE.Mesh[] = [];
  private readonly hazardGroup = new THREE.Group();
  private readonly effectPool: PooledEffect[];

  /** Last-seen `{respawns, position}` per food id, used only to detect a pickup edge. Never a live reference into simulation state — see `transforms.ts#detectFoodPickups`. */
  private readonly lastFoodRespawns = new Map<string, FoodRespawnRecord>();
  private hasAppliedFood = false;
  private hazardTouching: ReadonlySet<string> = new Set();
  /** The simulation tick the trails were last sampled at, so pushing a trail point tracks simulated motion, not render-frame rate (see `applyAgents`). */
  private lastTrailTick: number | undefined;
  private lastFrameTimeMs: number | undefined;
  private emaFps: number | undefined;
  private contextLost = false;
  private disposed = false;

  private readonly handleContextLost = createContextLossHandler(() => {
    this.contextLost = true;
    this.options.onContextLost?.({ reason: 'webglcontextlost' });
  });

  constructor(options: ArenaSceneOptions) {
    this.options = options;
    this.canvas = options.canvas;
    this.container = options.container ?? options.canvas.parentElement ?? options.canvas;
    this.reducedMotion = options.reducedMotion ?? false;
    this.config = options.arenaConfig ?? ARENA_CONFIG;

    try {
      this.renderer = new THREE.WebGLRenderer({ canvas: this.canvas, antialias: true, alpha: false });
    } catch (error) {
      throw new ArenaSceneUnavailableError(error);
    }
    // Everything past this point can construct DOM canvases, register
    // listeners, and build geometry — any of which can throw (e.g. an old
    // browser without CanvasRenderingContext2D#roundRect). Once the
    // WebGLRenderer exists it owns a real GPU context, so a throw here must
    // roll that back before propagating, or the context and listener leak
    // with no `ArenaScene` instance around to dispose them.
    let controls: OrbitControls | undefined;
    try {
      this.renderer.setPixelRatio(Math.min(window.devicePixelRatio ?? 1, 2));
      this.renderer.outputColorSpace = THREE.SRGBColorSpace;
      this.canvas.addEventListener('webglcontextlost', this.handleContextLost, false);

      this.scene.background = new THREE.Color('#070b13');
      this.scene.fog = new THREE.Fog('#070b13', this.config.halfWidth * 1.6, this.config.halfWidth * 5);

      const diagonal = Math.hypot(this.config.halfWidth, this.config.halfDepth);
      this.camera = new THREE.PerspectiveCamera(50, 1, 0.1, diagonal * 20);
      this.camera.position.set(diagonal * 0.9, diagonal * 1.1, diagonal * 1.3);

      // Assigned through a local first: if OrbitControls' own constructor is
      // what throws, `this.controls` is a readonly field TypeScript will not
      // let the catch block below read as possibly-unassigned, but a local
      // `let` can be read safely either way.
      controls = new OrbitControls(this.camera, this.renderer.domElement);
      this.controls = controls;
      this.controls.enableDamping = !this.reducedMotion;
      this.controls.dampingFactor = CONTROLS_DAMPING_FACTOR;
      this.controls.maxPolarAngle = Math.PI / 2 - 0.02;
      this.controls.minDistance = diagonal * 0.4;
      this.controls.maxDistance = diagonal * 6;
      this.controls.target.set(0, 0, 0);
      this.controls.update();

      this.buildLighting();
      this.buildFloorAndWalls();

      for (const id of ['left', 'right'] as const) {
        const { group, labelSprites } = buildAgentBody(id, this.config.agentRadius);
        this.scene.add(group);
        const trail = buildTrail(AGENT_SLOT_ACCENT[id]);
        this.scene.add(trail.line);
        this.agentVisuals.set(id, { group, labelSprites, trail });
      }

      this.foodGeometry = new THREE.SphereGeometry(this.config.foodRadius, 10, 8);
      this.foodMaterial = new THREE.MeshStandardMaterial({
        color: '#8fe3a6',
        emissive: '#123018',
        roughness: 0.4
      });
      this.scene.add(this.foodGroup);

      this.hazardGeometry = new THREE.IcosahedronGeometry(this.config.hazardRadius, 0);
      this.hazardMaterial = new THREE.MeshStandardMaterial({
        color: '#ef476f',
        emissive: '#3a0d18',
        roughness: 0.35,
        flatShading: true
      });
      this.scene.add(this.hazardGroup);

      this.effectPool = buildEffectPool();
      for (const effect of this.effectPool) this.scene.add(effect.sprite);

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

  private buildLighting(): void {
    const hemisphere = new THREE.HemisphereLight('#4a6a86', '#0d1420', 0.9);
    this.scene.add(hemisphere);
    const sun = new THREE.DirectionalLight('#eef4ff', 1.1);
    sun.position.set(this.config.halfWidth, this.config.halfWidth * 1.5, this.config.halfDepth);
    this.scene.add(sun);
  }

  private buildFloorAndWalls(): void {
    const width = this.config.halfWidth * 2;
    const depth = this.config.halfDepth * 2;
    const floor = new THREE.Mesh(
      new THREE.PlaneGeometry(width, depth),
      new THREE.MeshStandardMaterial({ color: '#0e1a28', roughness: 0.95 })
    );
    floor.rotation.x = -Math.PI / 2;
    this.scene.add(floor);

    const grid = new THREE.GridHelper(Math.max(width, depth), 20, '#1d2b3a', '#152230');
    grid.position.y = 0.005;
    this.scene.add(grid);

    const wallMaterial = new THREE.MeshStandardMaterial({
      color: '#2b4354',
      transparent: true,
      opacity: 0.4,
      roughness: 0.6
    });
    const walls: Array<[number, number, number, number, number]> = [
      // width, depth, x, y, z — one box per boundary edge
      [width + WALL_THICKNESS * 2, WALL_THICKNESS, 0, WALL_HEIGHT / 2, this.config.halfDepth],
      [width + WALL_THICKNESS * 2, WALL_THICKNESS, 0, WALL_HEIGHT / 2, -this.config.halfDepth],
      [WALL_THICKNESS, depth, this.config.halfWidth, WALL_HEIGHT / 2, 0],
      [WALL_THICKNESS, depth, -this.config.halfWidth, WALL_HEIGHT / 2, 0]
    ];
    for (const [boxWidth, boxDepth, x, y, z] of walls) {
      const wall = new THREE.Mesh(new THREE.BoxGeometry(boxWidth, WALL_HEIGHT, boxDepth), wallMaterial);
      wall.position.set(x, y, z);
      this.scene.add(wall);
    }
  }

  private setupResizeObserver(): void {
    this.resizeObserver = createCanvasResizeObserver(this.container, (width, height) => this.resize(width, height));
  }

  /**
   * Mirror a live `prefers-reduced-motion` change: toggles camera damping
   * immediately, and gates hazard spin / trail sampling / effect growth
   * animation from the next `update()` call on. The host is responsible for
   * subscribing to the media query's `change` event (see `App.svelte`) —
   * this class only reacts to the value it's given.
   */
  setReducedMotion(value: boolean): void {
    if (this.disposed || this.reducedMotion === value) return;
    this.reducedMotion = value;
    this.controls.enableDamping = !value;
  }

  /**
   * Switch which of `agentId`'s three preallocated label sprites is
   * `visible`, without creating any new canvas/texture/material — see
   * `buildAgentBody`'s doc comment. The body's shape/color never changes:
   * shape identifies the *slot* (left/right); this label is the only thing
   * that identifies *topology*.
   *
   * The host must call this once per agent right after construction (with
   * whichever topology is actually in effect at that point — see
   * `App.svelte`'s pairing of scene construction with its own `topology`
   * state) and again after every topology switch that has *already*
   * succeeded — never speculatively before a switch is confirmed, or the
   * label would claim a topology that isn't actually running yet.
   *
   * This is the fix for a real shipped honesty bug: an earlier revision of
   * this file hardcoded the left agent's label to "BIO" and the right
   * agent's to "REWIRED" for the lifetime of the scene, so switching either
   * arm's topology (e.g. to "Disconnected") left the 3D canvas still
   * displaying "BIO" above a slot that was, by then, provably running zero
   * edges — the most visually prominent surface in the whole demo silently
   * asserting biological provenance for an arm that had none. `setAgentTopology`
   * is the only thing standing between this scene and that bug recurring.
   */
  setAgentTopology(agentId: AgentId, mode: GraphMode): void {
    if (this.disposed) return;
    const visual = this.agentVisuals.get(agentId);
    if (!visual) return;
    for (const candidate of TOPOLOGY_MODES) {
      visual.labelSprites[candidate].visible = candidate === mode;
    }
  }

  /** Explicit resize hook, also used internally by the `ResizeObserver` callback. */
  resize(width: number, height: number): void {
    if (this.disposed) return;
    resizeRendererAndCamera(this.renderer, this.camera, width, height);
  }

  private ensurePoolSize(
    pool: THREE.Mesh[],
    group: THREE.Group,
    geometry: THREE.BufferGeometry,
    material: THREE.Material,
    count: number
  ): void {
    while (pool.length < count) {
      const mesh = new THREE.Mesh(geometry, material);
      mesh.visible = false;
      group.add(mesh);
      pool.push(mesh);
    }
  }

  /**
   * Place and scale every active pooled mesh at `entities[index]`'s
   * position — lifted onto the floor by, and scaled relative to, that
   * entity's own `radius` — and hide every remaining pool slot past
   * `entities.length`. `ensurePoolSize` already extracts the *growth* half
   * of pooling; this is the matching *sync* half `applyFood`/`applyHazards`
   * otherwise duplicate. Callers needing a per-mesh extra (e.g. hazard
   * spin) do it in their own follow-up `for` loop rather than a callback
   * here, so this stays allocation-free with no closure created per frame.
   */
  private syncPool<T extends { readonly position: Readonly<Vec2>; readonly radius: number }>(
    pool: readonly THREE.Mesh[],
    entities: readonly T[],
    baseRadius: number
  ): void {
    for (let index = 0; index < entities.length; index += 1) {
      const entity = entities[index];
      const mesh = pool[index];
      mesh.visible = true;
      mesh.scale.setScalar(entity.radius / baseRadius);
      mesh.position.set(entity.position.x, entity.radius, entity.position.z);
    }
    for (let index = entities.length; index < pool.length; index += 1) {
      pool[index].visible = false;
    }
  }

  private applyAgents(snapshot: Readonly<ArenaSnapshot>): void {
    // A tick moving backward only happens across a future Reset (not yet
    // wired up — see App.svelte's demo-loop comment). Clear every trail so
    // a new run never draws a line back to the previous run's positions.
    if (this.lastTrailTick !== undefined && snapshot.tick < this.lastTrailTick) {
      for (const visual of this.agentVisuals.values()) {
        visual.trail.filled = 0;
        visual.trail.geometry.setDrawRange(TRAIL_CAPACITY, 0);
      }
    }
    // Sample trails once per simulation tick rather than once per render
    // call: `update()` can be (and, while paused, will be) called many
    // times per tick with an unchanged snapshot, and sampling every call
    // would both collapse the trail's history into one repeated point and
    // make its visible time-span depend on the display refresh rate.
    const shouldSampleTrail = !this.reducedMotion && snapshot.tick !== this.lastTrailTick;

    for (const agent of snapshot.agents) {
      const visual = this.agentVisuals.get(agent.id);
      if (!visual) continue;
      const transform = agentTransform(agent, this.config.agentRadius);
      visual.group.position.set(transform.position.x, transform.position.y, transform.position.z);
      visual.group.rotation.y = transform.rotationY;

      if (shouldSampleTrail) {
        visual.trail.filled = pushTrailPoint(visual.trail.positions, TRAIL_CAPACITY, visual.trail.filled, {
          x: agent.position.x,
          y: TRAIL_LIFT,
          z: agent.position.z
        });
        const attribute = visual.trail.geometry.getAttribute('position') as THREE.BufferAttribute;
        attribute.needsUpdate = true;
        visual.trail.geometry.setDrawRange(TRAIL_CAPACITY - visual.trail.filled, visual.trail.filled);
      }
    }
    if (shouldSampleTrail) this.lastTrailTick = snapshot.tick;
  }

  private applyFood(foods: readonly FoodState[]): void {
    this.ensurePoolSize(this.foodPool, this.foodGroup, this.foodGeometry, this.foodMaterial, foods.length);
    this.syncPool(this.foodPool, foods, this.config.foodRadius);

    for (const event of detectFoodPickups(this.hasAppliedFood ? this.lastFoodRespawns : undefined, foods)) {
      this.spawnEffect({ x: event.position.x, y: this.config.foodRadius, z: event.position.z }, FOOD_EFFECT_COLOR);
    }
    // Mutate each existing record in place rather than replacing it: food
    // ids are fixed for the lifetime of a world (arena/world.ts creates
    // exactly `foodCount` foods up front and never adds/removes one — a
    // respawn only moves an existing id), so after the first frame this
    // loop allocates nothing at all instead of two objects per food, every
    // render frame.
    for (const food of foods) {
      const record = this.lastFoodRespawns.get(food.id);
      if (record) {
        record.respawns = food.respawns;
        record.position.x = food.position.x;
        record.position.z = food.position.z;
      } else {
        this.lastFoodRespawns.set(food.id, { respawns: food.respawns, position: { ...food.position } });
      }
    }
    this.hasAppliedFood = true;
  }

  private applyHazards(snapshot: Readonly<ArenaSnapshot>, frameMs: number): void {
    const hazards = snapshot.hazards;
    this.ensurePoolSize(
      this.hazardPool,
      this.hazardGroup,
      this.hazardGeometry,
      this.hazardMaterial,
      hazards.length
    );
    this.syncPool(this.hazardPool, hazards, this.config.hazardRadius);
    if (!this.reducedMotion) {
      const spinStep = (frameMs / 1000) * HAZARD_SPIN_RADIANS_PER_SECOND;
      for (let index = 0; index < hazards.length; index += 1) {
        const mesh = this.hazardPool[index];
        mesh.rotation.y = (mesh.rotation.y + spinStep) % TWO_PI;
      }
    }

    // Both agents and hazards must come from the *same* snapshot: comparing
    // this frame's hazards against a previous frame's agents would mix two
    // different interpolation instants and could show a contact that never
    // coexisted, or miss a real one (both reviewers on this bean caught the
    // earlier version of this bug).
    const contacts: HazardContactResult = detectHazardContacts(
      snapshot.agents,
      hazards,
      this.config.agentRadius,
      this.hazardTouching
    );
    this.hazardTouching = contacts.touching;
    for (const event of contacts.events) {
      this.spawnEffect(
        { x: event.position.x, y: this.config.hazardRadius, z: event.position.z },
        HAZARD_EFFECT_COLOR
      );
    }
  }

  /** Trigger a pooled burst effect. Reuses the oldest slot; never allocates a sprite/material. */
  private spawnEffect(position: { x: number; y: number; z: number }, color: THREE.Color): void {
    let target: PooledEffect | undefined;
    for (let index = 0; index < this.effectPool.length; index += 1) {
      if (this.effectPool[index].life <= 0) {
        target = this.effectPool[index];
        break;
      }
    }
    if (!target) {
      target = this.effectPool[0];
      for (let index = 1; index < this.effectPool.length; index += 1) {
        if (this.effectPool[index].life < target.life) target = this.effectPool[index];
      }
    }
    target.material.color.copy(color);
    target.sprite.position.set(position.x, position.y, position.z);
    target.sprite.visible = true;
    target.life = EFFECT_LIFE_SECONDS;
  }

  private updateEffects(frameMs: number): void {
    const dt = frameMs / 1000;
    const baseScale = this.config.agentRadius * EFFECT_BASE_SCALE_MULTIPLIER;
    for (const effect of this.effectPool) {
      if (effect.life <= 0) continue;
      effect.life = Math.max(0, effect.life - dt);
      const t = 1 - effect.life / EFFECT_LIFE_SECONDS;
      effect.material.opacity = 1 - t;
      const growth = this.reducedMotion ? 1 : 1 + t * EFFECT_MAX_GROWTH;
      effect.sprite.scale.setScalar(baseScale * growth);
      if (effect.life <= 0) effect.sprite.visible = false;
    }
  }

  /** Milliseconds since the previous `update()` call, from renderer/host frame timing only — never neural/simulated time. */
  private trackFrameTiming(nowMs: number): number {
    const rawFrameMs = this.lastFrameTimeMs === undefined ? 0 : nowMs - this.lastFrameTimeMs;
    this.lastFrameTimeMs = nowMs;
    // Clamp so a tab coming back from background/throttled state, or a
    // non-monotonic timestamp, cannot report a nonsensical one-off FPS
    // spike or drive a huge hazard-spin step.
    const frameMs = Math.min(Math.max(rawFrameMs, 0), MAX_FRAME_DELTA_MS);
    if (frameMs > 0) {
      // Guard the EMA seed too: two calls under 1ms apart would otherwise
      // seed `emaFps` with a four-digit spike that then takes many frames
      // to decay back toward a plausible value.
      const instantaneous = Math.min(1000 / frameMs, 240);
      this.emaFps = this.emaFps === undefined ? instantaneous : this.emaFps + (instantaneous - this.emaFps) * FPS_SMOOTHING;
      this.options.onFrame?.({ fps: this.emaFps, frameMs });
    }
    return frameMs;
  }

  /**
   * Render one frame from an already-interpolated, read-only snapshot
   * (see `arena/world.ts#createSnapshot`). Called by the host once per
   * animation frame; safe to call with an unchanged snapshot while the
   * simulation is paused, since rendering is decoupled from the fixed
   * simulation timestep — trail sampling and hazard/food event detection
   * are keyed off `snapshot.tick`/respawn counters, not the render cadence.
   * Never mutates `snapshot` or anything it points to.
   *
   * `nowMs` should be the same `requestAnimationFrame` timestamp the host
   * already has (e.g. from its own rAF callback); passing it avoids a
   * second, independent `performance.now()` clock that would otherwise
   * include this call's own `stepWorld`/`createSnapshot` time and jitter
   * the reported FPS relative to the display's vsync. Defaults to
   * `performance.now()` for standalone callers/tests.
   */
  update(snapshot: Readonly<ArenaSnapshot>, nowMs: number = performance.now()): void {
    if (this.disposed || this.contextLost) return;
    const frameMs = this.trackFrameTiming(nowMs);

    this.applyAgents(snapshot);
    this.applyFood(snapshot.foods);
    this.applyHazards(snapshot, frameMs);
    this.updateEffects(frameMs);

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
      // If the context is already lost (dispose() called from an
      // onContextLost handler), skip forceContextLoss() — it's a harmless
      // no-op there but some WebGL implementations log a spurious warning.
      skipForceContextLoss: this.contextLost,
      clearScene: true
    });
  }
}
