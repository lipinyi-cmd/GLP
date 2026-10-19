function loss = expand_kfoldLossSVM_time(optVars, X, Y)
% Mean RMSE from four expanding-window validation folds.
    k = 4;
    rng(42);
    N = size(X, 1);
    indices = round(linspace(1, N+1, k+2));
    rmse_all = zeros(k, 1);

    for i = 1:k
        test_idx = indices(i+1):indices(i+2)-1;
        train_idx = 1:test_idx(1)-1;

        X_train = X(train_idx, :);
        Y_train = Y(train_idx, :);
        X_test = X(test_idx, :);
        Y_test = Y(test_idx, :);

        rng(42);
        model = fitrsvm(X_train, Y_train, ...
            'KernelFunction', 'rbf', ...
            'BoxConstraint', optVars.BoxConstraint, ...
            'KernelScale', optVars.KernelScale);

        yPred = predict(model, X_test);
        rmse_all(i) = sqrt(mean((yPred - Y_test).^2));
    end

    loss = mean(rmse_all);
end
