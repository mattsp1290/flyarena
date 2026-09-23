import * as THREE from 'three';
import { describe, expect, it, vi } from 'vitest';
import { disposeObject3D, disposeRenderer } from '../../src/lib/render/dispose';

describe('disposeObject3D', () => {
  it('disposes the geometry and material for every mesh in the tree, including nested groups', () => {
    const root = new THREE.Group();
    const geometryA = new THREE.BoxGeometry(1, 1, 1);
    const materialA = new THREE.MeshBasicMaterial();
    root.add(new THREE.Mesh(geometryA, materialA));

    const nested = new THREE.Group();
    const geometryB = new THREE.SphereGeometry(1);
    const materialB = new THREE.MeshStandardMaterial();
    nested.add(new THREE.Mesh(geometryB, materialB));
    root.add(nested);

    const geometryDisposeA = vi.fn();
    const geometryDisposeB = vi.fn();
    const materialDisposeA = vi.fn();
    const materialDisposeB = vi.fn();
    geometryA.addEventListener('dispose', geometryDisposeA);
    geometryB.addEventListener('dispose', geometryDisposeB);
    materialA.addEventListener('dispose', materialDisposeA);
    materialB.addEventListener('dispose', materialDisposeB);

    disposeObject3D(root);

    expect(geometryDisposeA).toHaveBeenCalledTimes(1);
    expect(geometryDisposeB).toHaveBeenCalledTimes(1);
    expect(materialDisposeA).toHaveBeenCalledTimes(1);
    expect(materialDisposeB).toHaveBeenCalledTimes(1);
  });

  it('disposes every texture a material references', () => {
    const geometry = new THREE.PlaneGeometry(1, 1);
    const texture = new THREE.Texture();
    const material = new THREE.MeshStandardMaterial({ map: texture, emissiveMap: texture });
    const mesh = new THREE.Mesh(geometry, material);
    const textureDispose = vi.fn();
    texture.addEventListener('dispose', textureDispose);

    disposeObject3D(mesh);

    // The same texture is referenced from two material slots (map,
    // emissiveMap); Texture#dispose is idempotent, so being called twice is
    // expected and harmless.
    expect(textureDispose).toHaveBeenCalledTimes(2);
  });

  it('disposes each material of a multi-material mesh', () => {
    const geometry = new THREE.BoxGeometry(1, 1, 1);
    const materials = [new THREE.MeshBasicMaterial(), new THREE.MeshBasicMaterial()];
    const mesh = new THREE.Mesh(geometry, materials);
    const spies = materials.map((material) => {
      const spy = vi.fn();
      material.addEventListener('dispose', spy);
      return spy;
    });

    disposeObject3D(mesh);

    for (const spy of spies) expect(spy).toHaveBeenCalledTimes(1);
  });

  it('is safe to call on objects with no geometry or material', () => {
    const root = new THREE.Group();
    root.add(new THREE.Object3D(), new THREE.Group());
    expect(() => disposeObject3D(root)).not.toThrow();
  });
});

describe('disposeRenderer', () => {
  it('disposes controls, force-loses the context, then disposes the renderer, in that order', () => {
    const calls: string[] = [];
    const controls = { dispose: () => calls.push('controls') };
    const renderer = {
      dispose: () => calls.push('renderer'),
      forceContextLoss: () => calls.push('context-loss')
    };

    disposeRenderer(renderer, controls);

    expect(calls).toEqual(['controls', 'context-loss', 'renderer']);
  });

  it('tolerates a renderer without forceContextLoss and no controls argument', () => {
    const renderer = { dispose: vi.fn() };
    expect(() => disposeRenderer(renderer)).not.toThrow();
    expect(renderer.dispose).toHaveBeenCalledTimes(1);
  });

  it('skips forceContextLoss when the context is already known to be lost', () => {
    const renderer = { dispose: vi.fn(), forceContextLoss: vi.fn() };

    disposeRenderer(renderer, undefined, { skipForceContextLoss: true });

    expect(renderer.forceContextLoss).not.toHaveBeenCalled();
    expect(renderer.dispose).toHaveBeenCalledTimes(1);
  });
});
