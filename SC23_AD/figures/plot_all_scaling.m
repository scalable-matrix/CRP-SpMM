% n = 256
n_node = [8, 18, 32, 50, 72, 98, 128];

%%
amazon_crp      = [ 3.64,  2.18,  1.62,  1.10, 0.98, 1.00, 1.02];
amazon_comb_1da = [     , 11.00, 10.74,  9.07, 7.23, 3.80, 3.06];
amazon_comb_2da = [25.41, 15.34, 10.19, 10.48, 6.59, 5.55, 5.39];
amazon_comb_2dc = [13.50,  8.13,  4.46,  3.91, 3.30, 2.60, 2.35];
plot_scaling(n_node, amazon_comb_2dc, amazon_crp, 'Amazon');

%%
orkut_crp      = [ 1.77,  1.06,  0.74, 0.56, 0.49, 0.50, 0.49];
orkut_comb_1da = [20.08, 13.17, 11.24, 7.41, 5.12, 4.25, 5.86];
orkut_comb_2da = [16.38,  9.12,  6.30, 6.27, 4.20, 3.71, 5.17];
orkut_comb_2dc = [ 8.36,  5.29,  3.48, 3.09, 2.65, 2.30, 2.45];
plot_scaling(n_node, orkut_comb_2dc, orkut_crp, 'com-Orkut');

%%
nm7_crp      = [ 3.39,  1.90,  1.36,  1.02,  0.89, 0.92, 0.95];
nm7_comb_1da = [70.23, 30.65, 18.02, 13.26, 11.78, 9.46, 7.79];
nm7_comb_2da = [41.97, 17.95, 12.60,  9.46,  7.44, 6.66, 6.25];
nm7_comb_2dc = [22.61, 10.86,  9.28,  8.14,  7.25, 6.47, 6.26];
plot_scaling(n_node, nm7_comb_2dc, nm7_crp, 'nm7');

%% 
cage15_crp      = [ 1.30, 0.79, 0.51, 0.36, 0.37, 0.31, 0.35];
cage15_comb_1da = [13.84, 9.88, 7.32, 5.99, 4.42, 4.08, 3.53];
cage15_comb_2da = [11.26, 7.67, 5.80, 5.07, 4.48, 3.14, 2.96];
cage15_comb_2dc = [ 5.34, 4.04, 3.33, 2.89, 2.61, 2.42, 2.30];
plot_scaling(n_node, cage15_comb_2dc, cage15_crp, 'cage15');