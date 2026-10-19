%% Configuration
scriptDir = fileparts(mfilename('fullpath'));
addpath(scriptDir);
dataFile = fullfile(scriptDir, 'data', 'X_y_2.xlsx');
targetSheet = 'y_30860019';
numRows = 730;
useParallel = false;
assert(isfile(dataFile), 'Data file not found: %s. See README.md for data setup.', dataFile);

rng(42, 'twister');
tic;

%% Load data and add the previous water level
Rainfall = readmatrix(dataFile);
ll = readmatrix(dataFile, 'Sheet', targetSheet);
x = Rainfall(1:numRows, 1:6);
y = ll(1:numRows, 2);

F = 1;
lag1 = [NaN(1, F); y(1:end-1, :)];
x = [x, lag1];
validIdx = all(~isnan(x), 2);
x = x(validIdx, :);
y = y(validIdx);

targetSeries = y;
featureSeries = x;
windowSize = 7;
stride = 1;
tstep = 0;
numFeatures = size(featureSeries, 2);

%% Split chronologically and normalize with training statistics
numTotalSteps = size(featureSeries, 1);
splitIdx = floor(0.8 * numTotalSteps);
trainFeaturesRaw = featureSeries(1:splitIdx, :);
trainTargetsRaw = targetSeries(1:splitIdx, :);

% Include history so the first test target follows the training period.
testStartIdx = splitIdx - windowSize - tstep + 2;
testFeaturesRaw = featureSeries(testStartIdx:end, :);
testTargetsRaw = targetSeries(testStartIdx:end, :);

min_vals_f = min(trainFeaturesRaw, [], 1);
max_vals_f = max(trainFeaturesRaw, [], 1);
min_vals_t = min(trainTargetsRaw, [], 1);
max_vals_t = max(trainTargetsRaw, [], 1);

trainFeatures = (trainFeaturesRaw - min_vals_f) ./ (max_vals_f - min_vals_f);
trainTargets = (trainTargetsRaw - min_vals_t) ./ (max_vals_t - min_vals_t);
testFeatures = (testFeaturesRaw - min_vals_f) ./ (max_vals_f - min_vals_f);
testTargets = (testTargetsRaw - min_vals_t) ./ (max_vals_t - min_vals_t);

%% Create sequence windows
numTrainSamples = floor((size(trainFeatures, 1) - windowSize - tstep) / stride) + 1;
XTrain = zeros(windowSize, numFeatures, numTrainSamples);
YTrain = zeros(1, 1, numTrainSamples);
for i = 1:numTrainSamples
    idx = (i-1)*stride + 1 : (i-1)*stride + windowSize;
    XTrain(:, :, i) = trainFeatures(idx, :);
    YTrain(:, :, i) = trainTargets((i-1)*stride + windowSize + tstep, :);
end

numTestSamples = floor((size(testFeatures, 1) - windowSize - tstep) / stride) + 1;
XTest = zeros(windowSize, numFeatures, numTestSamples);
YTest = zeros(1, 1, numTestSamples);
for i = 1:numTestSamples
    idx = (i-1)*stride + 1 : (i-1)*stride + windowSize;
    XTest(:, :, i) = testFeatures(idx, :);
    YTest(:, :, i) = testTargets((i-1)*stride + windowSize + tstep, :);
end

XTrain = dlarray(XTrain, "TCB");
YTrain = dlarray(YTrain, "TCB");
XTest = dlarray(XTest, "TCB");
YTest = dlarray(YTest, "TCB");
rng(42);

%% Optimize hyperparameters with expanding-window validation
optimVars = [
    optimizableVariable('NumHiddenUnits', [20, 200], 'Type', 'integer')
    optimizableVariable('InitialLearnRate', [1e-4, 1e-1], 'Transform', 'log')
    optimizableVariable('L2Regularization', [1e-6, 1e-2], 'Transform', 'log')
    optimizableVariable('Dropout', [0, 0.5])
    optimizableVariable('MiniBatchSize', [16, 128], 'Type', 'integer')
];

rng(42);
ObjFcn = @(optVars) expand_T1_trainAndValidate_trainnet_w_K_Fold_times(optVars, XTrain, YTrain);
rng(42);
results = bayesopt(ObjFcn, optimVars, ...
    'MaxObjectiveEvaluations', 30, ...
    'AcquisitionFunctionName', 'expected-improvement-plus', ...
    'UseParallel', useParallel);

bestVars = results.XAtMinObjective;
fprintf('Best NumHiddenUnits: %d\n', bestVars.NumHiddenUnits);
fprintf('Best InitialLearnRate: %g\n', bestVars.InitialLearnRate);
fprintf('Best L2Regularization: %g\n', bestVars.L2Regularization);
fprintf('Best Dropout: %g\n', bestVars.Dropout);
fprintf('Best MiniBatchSize: %d\n', bestVars.MiniBatchSize);

%% Train the final model
YTrain = squeeze(YTrain);
layers = [
    sequenceInputLayer(size(featureSeries, 2))
    lstmLayer(bestVars.NumHiddenUnits, OutputMode="last", Name='lstm_1')
    dropoutLayer(bestVars.Dropout, Name='drop')
    fullyConnectedLayer(1, Name='fc')
];

options = trainingOptions('adam', ...
    'InitialLearnRate', bestVars.InitialLearnRate, ...
    'L2Regularization', bestVars.L2Regularization, ...
    'MaxEpochs', 100, ...
    'MiniBatchSize', bestVars.MiniBatchSize, ...
    'Shuffle', 'every-epoch', ...
    'Plots', 'training-progress', ...
    'Verbose', false, ...
    Metrics="rmse");
rng(42);
net = trainnet(XTrain, YTrain, layers, "mse", options);

%% Evaluate the training set in normalized units
Ytrain = predict(net, XTrain);
Ytrain = extractdata(Ytrain);
Ytrain = squeeze(Ytrain);
YTrain = extractdata(YTrain);
YTrain = squeeze(YTrain);

figure;
plot(YTrain(:), '-o', 'Color', [255 0 0]./255, 'LineWidth', 0.8, 'MarkerSize', 4, 'MarkerFaceColor', [255 0 0]./255);
hold on;
plot(Ytrain(:), '-s', 'Color', [0 0 0]./255, 'LineWidth', 0.8, 'MarkerSize', 5, 'MarkerFaceColor', [0 0 0]./255);
hold off;
legend(["Observed" "Predicted"]);
xlabel("Sample");
title("Training set");

SS_res = sum((YTrain(:) - Ytrain(:)).^2);
SS_tot = sum((YTrain(:) - mean(YTrain(:))).^2);
R_squared_tr = 1 - (SS_res / SS_tot);
disp(['Training R-squared = ', num2str(R_squared_tr)]);
rmse_T = sqrt(mean((Ytrain(:) - YTrain(:)).^2));
M_rmse_T = mean(rmse_T);
fprintf('Training RMSE: %.4f\n', M_rmse_T);

%% Evaluate the test set in normalized units
Ysimu = predict(net, XTest);
Ysimu = extractdata(Ysimu);
Ysimu = squeeze(Ysimu);
YTest = extractdata(YTest);
YTest = squeeze(YTest);

figure;
plot(YTest(:), '-o', 'Color', [255 0 0]./255, 'LineWidth', 0.8, 'MarkerSize', 4, 'MarkerFaceColor', [255 0 0]./255);
hold on;
plot(Ysimu(:), '-s', 'Color', [0 0 0]./255, 'LineWidth', 0.8, 'MarkerSize', 5, 'MarkerFaceColor', [0 0 0]./255);
hold off;
legend(["Observed" "Predicted"]);
xlabel("Sample");
title("Test set");

SS_res_te = sum((YTest(:) - Ysimu(:)).^2);
SS_tot_te = sum((YTest(:) - mean(YTest(:))).^2);
R_squared_te = 1 - (SS_res_te / SS_tot_te);
disp(['Test R-squared = ', num2str(R_squared_te)]);
rmse_test = sqrt(mean((Ysimu(:) - YTest(:)).^2));
M_rmse_test = mean(rmse_test);
fprintf('Test RMSE: %.4f\n', M_rmse_test);

%% Elapsed time
timeElapsed = toc;
disp(['Elapsed time: ', num2str(timeElapsed), ' seconds']);
t = seconds(timeElapsed);
t.Format = 'hh:mm:ss';
disp(t);
