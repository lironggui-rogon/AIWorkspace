import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # 1. 块级寻址：我是第几个“程序块”？
    pid = tl.program_id(axis=0)
    
    # 2. 张量块展开：生成当前块内所有元素的偏移量数组 [0, 1, ..., BLOCK_SIZE-1]
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # 3. 块级掩码：使用向量化的布尔数组进行边界保护
    mask = offsets < n_elements
    
    # 4. 块级访存与计算：一次性加载、计算并存回整个张量块
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)

def add(x: torch.Tensor, y: torch.Tensor):
    # 预先分配输出张量
    output = torch.empty_like(x)
    
    # 确保所有张量都在同一个正确的设备上
    assert x.is_cuda and y.is_cuda and output.is_cuda
    
    n_elements = output.numel()
    
    # 定义 SPMD 启动网格（Grid），决定并行运行的 Kernel 实例数量
    # 类似于 CUDA 中的 grid_size。使用 triton.cdiv 进行向上取整除法
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    
    # 启动 Kernel
    # 注意：传入的 torch.tensor 会被隐式转换为指向其首元素的指针
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    
    # 返回结果。注意：此时 GPU 计算是异步的，如果没有显式同步，返回时计算可能仍在进行
    return output

torch.manual_seed(0)
size = 98432 # 故意选择不能被 1024 整除的尺寸
x = torch.rand(size, device='cuda')
y = torch.rand(size, device='cuda')

output_torch = x + y
output_triton = add(x, y)

print(output_torch)
print(output_triton)
print(f'Torch 输出与 Triton 输出的最大差异为: '
      f'{torch.max(torch.abs(output_torch - output_triton))}')
# 如果输出差异为 0.0，则说明逻辑完全正确！

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],  # Argument names to use as an x-axis for the plot.
        x_vals=[2**i for i in range(12, 28, 1)],  # Different possible values for `x_name`.
        x_log=True,  # x axis is logarithmic.
        line_arg='provider',  # Argument name whose value corresponds to a different line in the plot.
        line_vals=['triton', 'torch'],  # Possible values for `line_arg`.
        line_names=['Triton', 'Torch'],  # Label name for the lines.
        styles=[('blue', '-'), ('green', '-')],  # Line styles.
        ylabel='GB/s',  # Label name for the y-axis.
        plot_name='vector-add-performance',  # Name for the plot. Used also as a file name for saving the plot.
        args={},  # Values for function arguments not in `x_names` and `y_name`.
    ))
def benchmark(size, provider):
    x = torch.rand(size, device='cuda', dtype=torch.float32)
    y = torch.rand(size, device='cuda', dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: x + y, quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: add(x, y), quantiles=quantiles)
    gbps = lambda ms: 3 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)

benchmark.run(print_data=True, show_plots=True, save_path='results')
