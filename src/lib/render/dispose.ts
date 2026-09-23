import type { Material, Object3D, Texture } from 'three';

/**
 * Every standard Three.js material property that can hold a disposable
 * texture. Enumerated rather than iterated dynamically so disposal is
 * predictable and does not depend on engine-internal property shapes.
 */
const TEXTURE_MATERIAL_KEYS = [
  'map',
  'alphaMap',
  'aoMap',
  'bumpMap',
  'displacementMap',
  'emissiveMap',
  'envMap',
  'gradientMap',
  'lightMap',
  'matcap',
  'metalnessMap',
  'normalMap',
  'roughnessMap',
  'specularMap',
  'specularIntensityMap',
  'specularColorMap',
  'clearcoatMap',
  'clearcoatNormalMap',
  'clearcoatRoughnessMap',
  'iridescenceMap',
  'iridescenceThicknessMap',
  'anisotropyMap',
  'sheenColorMap',
  'sheenRoughnessMap',
  'transmissionMap',
  'thicknessMap'
] as const;

const isTexture = (value: unknown): value is Texture =>
  typeof value === 'object' && value !== null && 'isTexture' in value && 'dispose' in value;

/** Dispose every texture a material references, then the material itself. */
const disposeMaterial = (material: Material): void => {
  const record = material as unknown as Record<string, unknown>;
  for (const key of TEXTURE_MATERIAL_KEYS) {
    const value = record[key];
    if (isTexture(value)) value.dispose();
  }
  material.dispose();
};

interface DisposableMesh {
  geometry?: { dispose?: () => void };
  material?: Material | Material[];
}

/**
 * Recursively free GPU-owned resources under `root`: geometries, materials,
 * and every texture a material references. Safe to call on plain groups and
 * on objects that hold no renderable geometry/material. Does not detach
 * `root` from its parent; callers that also want the node removed from the
 * scene graph should call `root.removeFromParent()` afterward.
 */
export const disposeObject3D = (root: Object3D): void => {
  root.traverse((child) => {
    const disposable = child as unknown as DisposableMesh;
    disposable.geometry?.dispose?.();
    const material = disposable.material;
    if (Array.isArray(material)) {
      material.forEach(disposeMaterial);
    } else if (material) {
      disposeMaterial(material);
    }
  });
};

interface DisposableRenderer {
  dispose: () => void;
  forceContextLoss?: () => void;
}

interface DisposableControls {
  dispose: () => void;
}

export interface DisposeRendererOptions {
  /**
   * Skip the `forceContextLoss()` call. Pass `true` when the context is
   * already lost (e.g. disposing from a `webglcontextlost` handler) —
   * forcing loss on an already-lost context is harmless but makes some
   * WebGL implementations log a spurious "extension not supported"-style
   * warning.
   */
  skipForceContextLoss?: boolean;
}

/**
 * Free the renderer's own GPU context/handles and any attached controls.
 * `forceContextLoss` is best-effort: not every renderer implementation
 * (including test doubles) exposes it.
 */
export const disposeRenderer = (
  renderer: DisposableRenderer,
  controls?: DisposableControls | undefined,
  options?: DisposeRendererOptions
): void => {
  controls?.dispose();
  if (!options?.skipForceContextLoss) renderer.forceContextLoss?.();
  renderer.dispose();
};
