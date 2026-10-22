export const NPU_ID_PREFIX = 'npu:';

export const npuAcceleratorId = (logicId: number): string => `${NPU_ID_PREFIX}${logicId}`;

export const isNpuAcceleratorId = (id: string): boolean => id.startsWith(NPU_ID_PREFIX);

/**
 * Job.gpu_ids is a comma-separated accelerator list. NVIDIA ids are plain
 * numbers (the pre-NPU format, unchanged); Ascend ids carry the `npu:` prefix.
 * `mps` is the Mac virtual device and keeps its legacy GPU 0 mapping.
 */
export function acceleratorIdsFromJob(gpuIds: string | null | undefined): string[] {
  if (!gpuIds) return [];
  if (gpuIds === 'mps') return ['0'];
  return gpuIds
    .split(',')
    .map(id => id.trim())
    .filter(Boolean);
}
