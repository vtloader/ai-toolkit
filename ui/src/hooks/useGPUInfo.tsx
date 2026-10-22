'use client';

import { AcceleratorOption, GpuInfo, NpuInfo } from '@/types';
import { useMemo } from 'react';
import useMonitorStream from '@/hooks/useMonitorStream';
import { npuAcceleratorId } from '@/utils/accelerators';

/**
 * Accelerator stats from the shared /api/monitor SSE stream. Data arrives live
 * every MONITOR_TICK_MS, so `reloadInterval` is accepted only for call-site
 * compatibility with the old polling implementation and is ignored.
 *
 * `gpuIds` filters both NVIDIA GPUs (plain index strings) and Ascend NPUs
 * (`npu:<logic id>`), matching the two id formats stored in Job.gpu_ids.
 */
export default function useGPUInfo(
  gpuIds: null | (number | string)[] = null,
  reloadInterval: null | number = null,
) {
  void reloadInterval;
  const { gpu, connected } = useMonitorStream();

  // Key on contents, not identity — call sites often build gpuIds inline.
  const gpuIdsKey = gpuIds ? gpuIds.join(',') : null;
  const ids = useMemo(
    () => (gpuIdsKey === null ? null : gpuIdsKey === '' ? [] : gpuIdsKey.split(',')),
    [gpuIdsKey],
  );

  const gpuList: GpuInfo[] = useMemo(() => {
    if (!gpu) return [];
    let gpus = [...gpu.gpus].sort((a, b) => a.index - b.index);
    if (ids !== null) {
      gpus = gpus.filter(g => ids.includes(`${g.index}`));
    }
    return gpus;
  }, [gpu, ids]);

  const npuList: NpuInfo[] = useMemo(() => {
    if (!gpu) return [];
    let npus = [...(gpu.npus ?? [])].sort((a, b) => a.index - b.index);
    if (ids !== null) {
      npus = npus.filter(n => ids.includes(npuAcceleratorId(n.logicId)));
    }
    return npus;
  }, [gpu, ids]);

  const acceleratorList: AcceleratorOption[] = useMemo(() => {
    if (!gpu) return [];
    const list: AcceleratorOption[] = [
      ...gpu.gpus.map(g => ({
        id: `${g.index}`,
        label: `GPU #${g.index} ${g.name}`,
        vendor: 'nvidia' as const,
      })),
      ...(gpu.npus ?? []).map(n => ({
        id: npuAcceleratorId(n.logicId),
        label: `NPU ${n.cardId}:${n.chipId} ${n.name}`,
        vendor: 'npu' as const,
      })),
    ];
    if (ids !== null) {
      return list.filter(a => ids.includes(a.id));
    }
    return list;
  }, [gpu, ids]);

  const isGPUInfoLoaded = gpu !== null;
  const status: 'idle' | 'loading' | 'success' | 'error' = gpu !== null ? 'success' : connected ? 'loading' : 'idle';

  return {
    gpuList,
    npuList,
    acceleratorList,
    isGPUInfoLoaded,
    status,
    // The stream is always live; nothing to refresh manually.
    refreshGpuInfo: async () => {},
  };
}
