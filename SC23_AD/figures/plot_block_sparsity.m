function plot_block_sparsity(A, bs)
    if (nargin < 2), bs = 1024; end
    [m, n] = size(A);
    nblk_m = ceil(m / bs);
    nblk_n = ceil(n / bs);
    blk_nnz = zeros(nblk_m, nblk_n);
    [row, col, ~] = find(A);
    for i = 1 : length(row)
        ib = ceil(row(i) / bs);
        jb = ceil(col(i) / bs);
        blk_nnz(ib, jb) = blk_nnz(ib, jb) + 1;
    end
    fig = figure('Render', 'painter', 'Position', [10 10 800 600]);
    log2_nnz = log2(blk_nnz + 2^(-5));
    imagesc(log2_nnz);
    xlabel_str = sprintf('Column block index (block size = %d)', bs);
    ylabel_str = sprintf('Row block index (block size = %d)', bs);
    xlabel(xlabel_str);
    ylabel(ylabel_str);
    font_size = 16;
    fig_handle = gca(fig);
    fig_handle.XAxis.FontSize = font_size;
    fig_handle.YAxis.FontSize = font_size;
    cb = colorbar;
    %cb.Label.Interpreter = 'latex';
    %cb.Label.FontSize = font_size;
    %cb.Label.String = '$\log_2(nnz + 2^{-5})$';
end