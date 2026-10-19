%% Data and experiment settings
scriptDir = fileparts(mfilename('fullpath'));
addpath(scriptDir);
dataFile = fullfile(scriptDir, 'data', 'X_y_2.xlsx');
targetSheet = 'y_30860017';
numRows = 2191;
trainRatio = 0.8;
trainWindow = 730;
useParallel = false;
assert(isfile(dataFile), 'Data file not found: %s. See README.md.', dataFile);

Rainfall = readtable(dataFile, 'Sheet', 'X', 'HeaderLines', 1);
ll = readtable(dataFile, 'Sheet', targetSheet, 'HeaderLines', 1);
X = table2array(Rainfall(1:numRows, 1:6));
y = table2array(ll(1:numRows, 2));

%% Lagged features
% Columns 1, 3, 4 and 5 represent radiation, humidity, precipitation and ET.
xRn = X(:, 1);
xReh = X(:, 3);
X_lag1 = lagmatrix(xReh, 7);
X_lag2 = lagmatrix(xRn, 1);
X_lag3 = lagmatrix(xReh, 2);
Y_lag1 = lagmatrix(y, 1);
Y_lag2 = lagmatrix(y, 2);
X_full = [X(:, 4), X(:, 5), Y_lag1, Y_lag2, X_lag1, X_lag2, X_lag3];
Y = y;

validIdx = 8:length(y);
X_full = X_full(validIdx, :);
Y = Y(validIdx);

%% Chronological split and normalization
nSamples = numel(Y);
nTrain = floor(trainRatio * nSamples);
Xtrain = X_full(1:nTrain, :);
Ytrain = Y(1:nTrain);
Xtest = X_full(nTrain+1:end, :);
Ytest = Y(nTrain+1:end);

% Fit normalization on the full training partition before selecting its tail.
min_vals = min(Xtrain);
max_vals = max(Xtrain);
Xtrain = (Xtrain - min_vals) ./ (max_vals - min_vals);
Xtest = (Xtest - min_vals) ./ (max_vals - min_vals);

min_vals_t = min(Ytrain);
max_vals_t = max(Ytrain);
Ytrain = (Ytrain - min_vals_t) ./ (max_vals_t - min_vals_t);
Ytest = (Ytest - min_vals_t) ./ (max_vals_t - min_vals_t);

% Retain the original experiment's last 730 training observations.
assert(nTrain >= trainWindow, 'Training partition must contain at least %d samples.', trainWindow);
trainIdx = nTrain-trainWindow+1:nTrain;
Xtrain = Xtrain(trainIdx, :);
Ytrain = Ytrain(trainIdx, :);

%% Bayesian optimization with expanding-window validation
rng(42);
optimVars = [
    optimizableVariable('BoxConstraint', [1e-2, 1e2], 'Transform', 'none');
    optimizableVariable('KernelScale', [1e-2, 1e2], 'Transform', 'none');
];
ObjFcn = @(optVars)expand_kfoldLossSVM_time(optVars, Xtrain, Ytrain);
results = bayesopt(ObjFcn, optimVars, ...
    'AcquisitionFunctionName', 'expected-improvement-plus', ...
    'MaxObjectiveEvaluations', 30, ...
    'Verbose', 1, ...
    'PlotFcn', {@plotObjectiveModel, @plotMinObjective}, ...
    'UseParallel', useParallel);
bestParams = results.XAtMinObjective;

%% Final model and evaluation
svmModel = fitrsvm(Xtrain, Ytrain, ...
    'KernelFunction', 'rbf', ...
    'BoxConstraint', bestParams.BoxConstraint, ...
    'KernelScale', bestParams.KernelScale);
fprintf('最佳超参数：BoxConstraint=%.4f, KernelScale=%.4f\n', ...
    bestParams.BoxConstraint, bestParams.KernelScale);

Ypred = predict(svmModel, Xtest);
mseTest = mean((Ytest - Ypred).^2);
fprintf('测试集 MSE: %.4f\n', mseTest);

figure;
plot(validIdx(nTrain+1:end), Ytest, '-o', 'DisplayName', '真实值');
hold on;
plot(validIdx(nTrain+1:end), Ypred, '-x', 'DisplayName', '预测值');
xlabel('时间索引');
ylabel('归一化水位');
title('测试集：真实 vs SVM 预测');
legend;
grid on;

figure;
Ypredt = predict(svmModel, Xtrain);
plot(validIdx(trainIdx), Ytrain, '-o', 'DisplayName', '真实值');
hold on;
plot(validIdx(trainIdx), Ypredt, '-x', 'DisplayName', '预测值');
xlabel('时间索引');
ylabel('归一化水位');
title('训练集：真实 vs SVM 预测');
legend;
grid on;
