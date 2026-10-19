% Run LSTM.m or MHSA_LSTM.m first to create predictions and scaling values.
assert(exist('YTrain', 'var') == 1 && exist('YTest', 'var') == 1 && ...
    exist('Ytrain', 'var') == 1 && exist('Ysimu', 'var') == 1 && ...
    exist('min_vals_t', 'var') == 1 && exist('max_vals_t', 'var') == 1, ...
    'Run LSTM.m or MHSA_LSTM.m before running figure6.m.');

%% Restore original target units
yt = YTrain(:) .* (max_vals_t - min_vals_t) + min_vals_t;
ytest = YTest(:) .* (max_vals_t - min_vals_t) + min_vals_t;
ytt = Ytrain(:) .* (max_vals_t - min_vals_t) + min_vals_t;
ytete = Ysimu(:) .* (max_vals_t - min_vals_t) + min_vals_t;

%% Full-series predictions and training metrics
figure;
yy = [yt; ytest];
yyv = [ytt; ytete];
plot(yy);
hold on;
plot(yyv);
legend(["Observation" "Simulation"]);
xlabel('Sample');
title('Full dataset');

SS_res_v = sum((yt - ytt).^2);
SS_tot_v = sum((yt - mean(yt)).^2);
R_squared_v = 1 - (SS_res_v / SS_tot_v);
fprintf('Train R^2: %.4f\n', R_squared_v);
rmse_v = sqrt(mean((yt - ytt).^2));
M_rmse_v = mean(rmse_v);
fprintf('Train RMSE: %.4f\n', M_rmse_v);
MAE = mean(abs(yt - ytt));
fprintf('Train MAE: %.4f\n', MAE);

%% Test predictions and metrics
figure;
plot(ytest);
hold on;
plot(ytete);
legend(["Observation" "Simulation"]);
xlabel('Sample');
title('Test dataset');

SS_res_v = sum((ytest - ytete).^2);
SS_tot_v = sum((ytest - mean(ytest)).^2);
R_squared_v = 1 - (SS_res_v / SS_tot_v);
fprintf('Test R^2: %.4f\n', R_squared_v);
rmse_v = sqrt(mean((ytest - ytete).^2));
M_rmse_v = mean(rmse_v);
fprintf('Test RMSE: %.4f\n', M_rmse_v);
MAE = mean(abs(ytest - ytete));
fprintf('Test MAE: %.4f\n', MAE);

R = corrcoef(ytest, ytete);
R = R(1, 2);
sd_obs = std(ytest);
sd1 = std(ytete);
nsd1 = sd1 / sd_obs;
nRMSE = M_rmse_v / sd_obs;
crmse = sqrt(mean(((ytest - mean(ytest)) - (ytete - mean(ytete))).^2));
crmse_n = crmse / std(ytest);
crmse_n_1 = sqrt(1 + nsd1^2 - 2 * nsd1 * R);
bias = mean(ytete - ytest);
fprintf('Test crmse_n: %.4f\n', crmse_n);
fprintf('Test crmse_n_1: %.4f\n', crmse_n_1);
fprintf('Test R: %.4f\n', R);
fprintf('Test bias: %.4f\n', bias);
fprintf('Test nRMSE: %.4f\n', nRMSE);
fprintf('Test nsd1: %.4f\n', nsd1);
RMSD = std(ytest - ytete, 1);
fprintf('Test RMSD: %.4f\n', RMSD);
STD = std(ytest, 1);
fprintf('Test STD: %.4f\n', STD);

%% Observed versus simulated values
figure('Color', 'w');
scatter(yt, ytt, 25, '+', 'MarkerEdgeColor', [0.4, 0.6, 0.9], 'DisplayName', 'Train data');
hold on;
p_train = polyfit(yt, ytt, 1);
x_train_fit = linspace(min(yt), max(yt), 100);
y_train_fit = polyval(p_train, x_train_fit);
plot(x_train_fit, y_train_fit, 'Color', [0.1, 0.3, 0.7], 'LineWidth', 2, 'DisplayName', 'Train fitted curve');

scatter(ytest, ytete, 25, '+', 'MarkerEdgeColor', [0.9, 0.5, 0.5], 'DisplayName', 'Test data');
p_test = polyfit(ytest, ytete, 1);
x_test_fit = linspace(min(ytest), max(ytest), 100);
y_test_fit = polyval(p_test, x_test_fit);
plot(x_test_fit, y_test_fit, 'Color', [0.8, 0.1, 0.1], 'LineWidth', 2, 'DisplayName', 'Test fitted curve');

xlabel('Observation (m)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Simulation (m)', 'FontSize', 12, 'FontWeight', 'bold');
legend('Location', 'southeast', 'Box', 'off');
grid off;
box on;
set(gca, 'LineWidth', 1.2, 'FontSize', 11);

%% Residuals
figure('Color', 'w');
res_train = ytt - yt;
res_test = ytete - ytest;
scatter(yt, res_train, 25, '+', 'MarkerEdgeColor', [0.4, 0.6, 0.9], 'DisplayName', 'Train residual');
hold on;
scatter(ytest, res_test, 25, '+', 'MarkerEdgeColor', [0.9, 0.5, 0.5], 'DisplayName', 'Test residual');

x_min = min([yt; ytest]);
x_max = max([yt; ytest]);
plot([x_min, x_max], [0, 0], 'k--', 'LineWidth', 1.5, 'DisplayName', 'Zero error line');

xlabel('Observation (m)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Residual (m)', 'FontSize', 12, 'FontWeight', 'bold');
title('Residual Plot', 'FontSize', 13, 'FontWeight', 'bold');
legend('Location', 'best', 'Box', 'off');
grid on;
box on;
set(gca, 'LineWidth', 1.2, 'FontSize', 11);
