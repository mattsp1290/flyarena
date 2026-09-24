import { readoutParameterCount, type ReadoutWeights } from './readout';

/** Portable flat layout shared by offline training and browser experiments. */
export function readoutFromFlat(
  theta: readonly number[] | Float32Array,
  inputSize: number,
  hiddenSize: number
): ReadoutWeights {
  if (
    !Number.isInteger(inputSize) ||
    inputSize < 1 ||
    inputSize > 4096 ||
    !Number.isInteger(hiddenSize) ||
    hiddenSize < 1 ||
    hiddenSize > 64 ||
    theta.length !== readoutParameterCount(inputSize, hiddenSize) ||
    Array.from(theta).some((v) => !Number.isFinite(v))
  ) {
    throw new Error('Invalid flat readout');
  }
  let cursor = 0;
  const take = (length: number) => {
    const values = Float32Array.from(theta.slice(cursor, cursor + length));
    cursor += length;
    if (values.some((v) => !Number.isFinite(v))) throw new Error('Readout exceeds float32 range');
    return values;
  };
  return {
    inputSize,
    hiddenSize,
    w1: take(hiddenSize * inputSize),
    b1: take(hiddenSize),
    w2: take(3 * hiddenSize),
    b2: take(3)
  };
}
