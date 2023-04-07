% 72 nodes, 144 procs
n = [128, 256, 512, 1024];

amazon_crp      = [0.68, 0.98,  1.57,  2.29];
amazon_comb_1da = [2.13, 7.23,  9.14, 16.45];
amazon_comb_2da = [3.25, 6.59, 14.70, 31.59];
amazon_comb_2dc = [2.25, 3.30,  6.14, 10.38];

orkut_crp      = [0.26, 0.49,  0.60,  1.02];
orkut_comb_1da = [1.97, 5.12, 16.47, 35.75];
orkut_comb_2da = [2.28, 4.20,  7.49, 17.10];
orkut_comb_2dc = [2.18, 2.65,  3.95,  6.83];

nm7_crp      = [0.62,  0.95,  1.39,  2.06];
nm7_comb_1da = [3.69, 11.78, 34.38, 65.06];
nm7_comb_2da = [4.85,  7.44, 13.83, 25.69];
nm7_comb_2dc = [6.05,  7.25, 11.23, 18.93];

cage15_crp      = [0.24, 0.37, 0.47,  0.85];
cage15_comb_1da = [1.70, 4.22, 9.33, 22.91];
cage15_comb_2da = [2.39, 4.48, 7.51, 12.29];
cage15_comb_2dc = [2.07, 2.61, 4.35,  6.22];

%% 
font_size = 14;
fig1 = figure('Renderer', 'painters', 'Position', [10 10 800 600]);
X = categorical({'CRP', 'CB-2D-C'});
X = reordercats(X, {'CRP', 'CB-2D-C'});

width  = 0.18;
height = 0.75;
bottom = 0.12;
lefts  = 0.06 : 0.25 : 1;
bars   = cell(4, 1);

%%
subplot('Position', [lefts(1) bottom width height]);
orkut_times = [orkut_crp; orkut_comb_2dc];
bars{1} = bar(X, orkut_times); grid on
ylabel('Runtime (seconds)', 'FontSize', font_size);
title('com-Orkut', 'FontSize', font_size);
set(gca, 'FontSize', font_size);

%%
subplot('Position', [lefts(2) bottom width height]);
nm7_times = [nm7_crp; nm7_comb_2dc];
bars{2} = bar(X, nm7_times); grid on
title('nm7', 'FontSize', font_size);
set(gca, 'FontSize', font_size);

%%
subplot('Position', [lefts(3) bottom width height]);
cage15_times = [cage15_crp; cage15_comb_2dc];
bars{3} = bar(X, cage15_times); grid on
title('cage15', 'FontSize', font_size);
set(gca, 'FontSize', font_size);

%%
subplot('Position', [lefts(4) bottom width height]);
amazon_times = [amazon_crp; amazon_comb_2dc];
bars{4} = bar(X, amazon_times); grid on
title('Amazon', 'FontSize', font_size);
set(gca, 'FontSize', font_size);

legend({'n = 128', 'n = 256', 'n = 512', 'n = 1024'}, ...
       'interpreter', 'latex', 'FontSize', font_size, ...
       'Location', 'north', 'orientation', 'horizon');

%% Make me happy
%{
sh_colors = [
134, 148, 173;
180, 191, 208;
 81, 124, 137;
 35,  43,  54;
] ./ 255;
for i = 1 : 4
    for j = 1 : 4
        bars{i}(j).FaceColor = sh_colors(j, :);
    end
end
%}