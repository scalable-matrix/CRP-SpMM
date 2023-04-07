function plot_scaling(n_node, comb_runtime, crp_runtime, title_str)
    fig1 = figure('Render', 'painter', 'Position', [10 10 800 600]);
    linscale = comb_runtime(1) * n_node(1) ./ n_node;
    loglog(n_node, comb_runtime, 'r-*'), hold on
    loglog(n_node, crp_runtime, 'b-o'), hold on
    loglog(n_node, linscale, 'k--'), hold on
    grid on;
    max_t = max(max(comb_runtime(:)), max(crp_runtime(:)));
    min_t = min(min(comb_runtime(:)), min(crp_runtime(:)));
    axis([min(n_node) * 0.9, max(n_node) / 0.9, min_t * 0.9, max_t / 0.9]);
    fig_handle = gca(fig1);
    font_size = 16;
    fig_handle.XAxis.FontSize = font_size;
    fig_handle.YAxis.FontSize = font_size;
    xticks([8, 18, 32, 50, 72, 98, 128]);
    yticks([0.5, 1, 2, 4, 8, 12, 16, 20]);
    xticklabels({'8', '18', '32', '50', '72', '98', '128'});
    yticklabels({'0.5', '1', '2', '4', '8', '12', '16', '20'});
    xlabel('Number of Nodes', 'FontSize', font_size);
    ylabel('Runtime (seconds)', 'FontSize', font_size);
    legend({'CombBLAS Best', 'CRP-SpMM', 'Linear Scaling'}, 'FontSize', font_size);
    %title(title_str);
end