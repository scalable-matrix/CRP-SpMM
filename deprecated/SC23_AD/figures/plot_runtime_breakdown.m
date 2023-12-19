% 128 nodes, 256 procs, n = 256
% Use the "average" column 
% CRP-SpMM timings: [copy A, copy B, local SpMM, redist]
% CombBLAS timings: [copy A, copy B, local SpMM, other]

crp_amazon  = [0.335, 0.380, 0.086, 0.026 + 0.078 + 0.093];
crp_orkut   = [0.166, 0.155, 0.040, 0.012 + 0.056 + 0.044];
crp_nm7     = [0.202, 0.462, 0.062, 0.029 + 0.061 + 0.109];
crp_cage15  = [0.075, 0.161, 0.007, 0.009 + 0.039 + 0.051];

comb_amazon = [0.473, 0.916, 0.919, 0];
comb_orkut  = [0.396, 1.611, 0.398, 0];
comb_nm7    = [0.802, 4.560, 0.693, 0];
comb_cage15 = [0.197, 1.832, 0.180, 0];

%% 
font_size = 14;
fig1 = figure('Renderer', 'painters', 'Position', [10 10 800 600]);
X = categorical({'CRP', 'CB-2D-C'});
X = reordercats(X, {'CRP', 'CB-2D-C'});

width  = 0.17;
height = 0.75;
bottom = 0.12;
lefts  = 0.075 : 0.25 : 1;
bars   = cell(4, 1);

%%
subplot('Position', [lefts(1) bottom width height]);
orkut_times = [crp_orkut; comb_orkut];
bars{1} = bar(X, orkut_times, 'stacked'); grid on
ylabel('Runtime (seconds)', 'FontSize', font_size);
title('com-Orkut', 'FontSize', font_size);
set(gca, 'FontSize', font_size);

%%
subplot('Position', [lefts(2) bottom width height]);
nm7_times = [crp_nm7; comb_nm7];
bars{2} = bar(X, nm7_times, 'stacked'); grid on
title('nm7', 'FontSize', font_size);
set(gca, 'FontSize', font_size);

%%
subplot('Position', [lefts(3) bottom width height]);
cage15_times = [crp_cage15; comb_cage15];
bars{3} = bar(X, cage15_times, 'stacked'); grid on
title('cage15', 'FontSize', font_size);
set(gca, 'FontSize', font_size);

%%
subplot('Position', [lefts(4) bottom width height]);
amazon_times = [crp_amazon; comb_amazon];
bars{4} = bar(X, amazon_times, 'stacked'); grid on
title('Amazon', 'FontSize', font_size);
set(gca, 'FontSize', font_size);

legend({'Replicate $A$', 'Replicate $B$', 'Local SpMM', 'Redist. $ABC$'}, ...
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