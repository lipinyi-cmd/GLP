function avg_rmse = expand_T1_trainAndValidate_trainnet_w_K_Fold_times(optVars, XTrain, YTrain)
% Mean RMSE from four expanding-window validation folds.
    k = 4;
    rng(42);
    % Labeled dlarray stores the TCB input in canonical CBT order.
    N = size(XTrain, 2);
    indices = round(linspace(1, N+1, k+2));
    numFeatures = size(XTrain, finddim(XTrain, 'C'));
    YTrain = squeeze(YTrain);
    rmse_all = zeros(k, 1);

    for i = 1:k
        test_idx = indices(i+1):indices(i+2)-1;
        train_idx = 1:test_idx(1)-1;

        X_train = XTrain(:, train_idx, :);
        Y_train = YTrain(:, train_idx, :);
        X_test = XTrain(:, test_idx, :);
        Y_test = YTrain(:, test_idx, :);

        rng(42);
        layers = [
            sequenceInputLayer(numFeatures)
            lstmLayer(optVars.NumHiddenUnits, 'OutputMode', 'last')
            dropoutLayer(optVars.Dropout)
            fullyConnectedLayer(1)
        ];

        rng(42, 'twister');
        options = trainingOptions('adam', ...
            'ExecutionEnvironment', 'cpu', ...
            'InitialLearnRate', optVars.InitialLearnRate, ...
            'L2Regularization', optVars.L2Regularization, ...
            'MaxEpochs', 100, ...
            'MiniBatchSize', optVars.MiniBatchSize, ...
            'Shuffle', 'once', ...
            'Verbose', false);

        rng(42);
        net = trainnet(X_train, Y_train, layers, "mse", options);

        Y_pred = predict(net, X_test);
        Y_pred = extractdata(Y_pred);
        Y_pred = squeeze(Y_pred);
        Y_test = extractdata(Y_test);
        Y_test = squeeze(Y_test);
        rmse_all(i) = sqrt(mean((Y_test - Y_pred).^2));
    end

    avg_rmse = mean(rmse_all);
end
