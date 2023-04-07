% 32 node, 64 procs
n = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048];

cage15_pn     = [1, 1, 2, 4, 4, 8, 8, 16, 16, 32, 32, 64];
cage15cm_pn   = [1, 1, 1, 1, 2, 2, 4,  4,  8,  8, 16, 32];
cage15_time   = [0.025, 0.038, 0.043, 0.065, 0.09, 0.15, 0.21, 0.34, 0.51, 0.83, 1.29, 1.82];
cage15cm_time = [0.017, 0.018, 0.030, 0.033, 0.05, 0.07, 0.12, 0.21, 0.33, 0.57, 1.10, 2.06];

font_size = 16;
fig1 = figure('Render', 'painter', 'Position', [10 10 800 600]);


colororder({'#0072BD', '#D95319'})
yyaxis left 
linscale = cage15cm_time(1) .* n; 
loglog(n, cage15_time, '-o'), hold on
loglog(n, cage15cm_time, '-d'), hold on
loglog(n, linscale, 'k--'), hold on
xticks(n), hold on
yticks([0.01 .* 2.^(0 : 9)]), hold on
axis([1 * 0.8, 2048 / 0.8, 0.01, 2.4]);
xlabel('Number of $B$ matrix columns ($n$)', 'Interpreter', 'latex', 'FontSize', font_size)
ylabel('Runtime (seconds)', 'FontSize', font_size);

yyaxis right
p3 = loglog(n, cage15_pn, '--o'); hold on
p4 = loglog(n, cage15cm_pn, '--d'); hold on
yticks([2.^(0 : 9)]), hold on
xticks(n), hold on
ylabel('Process Grid Size $p_n$ ($p_m = 64 / p_n$)', 'Interpreter', 'latex', 'FontSize', font_size);
grid on
fig1.CurrentAxes.YScale = 'log';
hold off
legend({'cage15 Runtime', 'cage15-cm Runtime', 'Linear Scaling', ...
        'cage15 $p_n$', 'cage15-cm $p_n$'}, 'Interpreter', 'latex', ...
        'FontSize', font_size, 'Location', 'northwest')