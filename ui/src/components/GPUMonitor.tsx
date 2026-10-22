import React, { useMemo } from 'react';
import Loading from '@/components/Loading';
import GPUWidget from '@/components/GPUWidget';
import NPUWidget from '@/components/NPUWidget';
import useMonitorStream from '@/hooks/useMonitorStream';

const GpuMonitor: React.FC = () => {
  // Live samples arrive every MONITOR_TICK_MS over the shared /api/monitor SSE stream
  const { gpu: gpuData, lastUpdated } = useMonitorStream();
  const loading = gpuData === null;
  const gpus = useMemo(() => gpuData?.gpus ?? [], [gpuData]);
  const npus = useMemo(() => gpuData?.npus ?? [], [gpuData]);
  const hasGpus = gpus.length > 0;
  const hasNpus = npus.length > 0;

  const getGridClasses = (gpuCount: number): string => {
    switch (gpuCount) {
      case 1:
        return 'grid-cols-1';
      case 2:
        return 'grid-cols-1 sm:grid-cols-2';
      case 3:
        return 'grid-cols-1 sm:grid-cols-2 lg:grid-cols-3';
      case 4:
        return 'grid-cols-1 sm:grid-cols-2 lg:grid-cols-4';
      case 5:
      case 6:
        return 'grid-cols-1 sm:grid-cols-2 lg:grid-cols-3';
      case 7:
      case 8:
        return 'grid-cols-1 sm:grid-cols-2 lg:grid-cols-4';
      case 9:
      case 10:
        return 'grid-cols-1 sm:grid-cols-2 lg:grid-cols-5';
      default:
        return 'grid-cols-1 sm:grid-cols-2 lg:grid-cols-3';
    }
  };

  const content = useMemo(() => {
    if (loading && !gpuData) {
      return <Loading />;
    }

    if (!gpuData) {
      return (
        <div className="bg-yellow-900 border border-yellow-700 text-yellow-300 px-4 py-3 rounded relative" role="alert">
          <span className="block sm:inline">No accelerator data available.</span>
        </div>
      );
    }

    if (!hasGpus && !hasNpus) {
      return (
        <div className="bg-yellow-900 border border-yellow-700 text-yellow-300 px-4 py-3 rounded relative" role="alert">
          <strong className="font-bold">No NVIDIA GPU or Ascend NPU detected!</strong>
          <span className="block sm:inline">
            {' '}
            Neither nvidia-smi nor npu-smi reported a usable device on this system.
          </span>
          {gpuData.error && <p className="mt-2 text-sm">{gpuData.error}</p>}
          {gpuData.npuError && <p className="mt-2 text-sm">{gpuData.npuError}</p>}
        </div>
      );
    }

    const gridClass = getGridClasses(Math.max(gpus.length, npus.length, 1));

    return (
      <div className="space-y-4">
        {hasGpus && (
          <div>
            {hasNpus && <h2 className="text-sm text-gray-400 mb-2">NVIDIA GPU</h2>}
            <div className={`grid ${gridClass} gap-3`}>
              {gpus.map((gpu, idx) => (
                <GPUWidget key={`gpu-${gpu.index}-${idx}`} gpu={gpu} />
              ))}
            </div>
          </div>
        )}
        {hasNpus && (
          <div>
            {hasGpus && <h2 className="text-sm text-gray-400 mb-2">Ascend NPU</h2>}
            <div className={`grid ${gridClass} gap-3`}>
              {npus.map((npu, idx) => (
                <NPUWidget key={`npu-${npu.cardId}-${npu.chipId}-${idx}`} npu={npu} />
              ))}
            </div>
          </div>
        )}
      </div>
    );
  }, [loading, gpuData, hasGpus, hasNpus, gpus, npus]);

  const title = hasNpus && !hasGpus ? 'NPU Monitor' : hasGpus && !hasNpus ? 'GPU Monitor' : 'Accelerator Monitor';

  return (
    <div className="w-full">
      <div className="flex justify-between items-center mb-2">
        <h1 className="text-md">{title}</h1>
        <div className="text-xs text-gray-500">Last updated: {lastUpdated?.toLocaleTimeString()}</div>
      </div>
      {content}
    </div>
  );
};

export default GpuMonitor;
