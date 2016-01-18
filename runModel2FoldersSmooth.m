clear all;

trainPath =  'C:\Users\Carlos\Documents\Projetos Machine Learning\Multiagent\Data\Input\*\*\';
testPath =  'C:\Users\Carlos\Documents\Projetos Machine Learning\Multiagent\Data\Test\*\*\';

file_pattern = '*[AVG-30].txt';
window_size = 8;
nclusters = 8;
hidden_layer = 6;
iterations = 30;

% Read train files
files = rdir(strcat(trainPath, file_pattern));
train_raw_data = [];

for i = 1 : size(files)
    x = importdata(files(i).name,'\t', 1);
    train_raw_data = [train_raw_data; x.data(:,1)'];
    clear x;
end

% Read test files
files = rdir(strcat(testPath, file_pattern));
test_raw_data = [];

for i = 1 : size(files)
    x = importdata(files(i).name,'\t', 1);
    test_raw_data = [test_raw_data; x.data(:,1)'];
    clear x;
end

raw_data = [train_raw_data;test_raw_data];

% clear raw_data;

smooth_data = reshape(smooth(raw_data), [], size(raw_data, 2));
detrend_data = raw_data - smooth_data;

% Normalize data
max_data = max(max(detrend_data));
min_data = min(min(detrend_data));
normalized_data = -1 + 2.*(detrend_data - min_data)./(max_data - min_data);

% Run Iterations
mean_elm_psa_rmse = zeros(iterations, 1);
mean_elm_psa_percent = zeros(iterations, 1);
mean_elm_rmse = zeros(iterations, 1);
mean_elm_percent = zeros(iterations, 1);
mean_arima_rmse = zeros(iterations, 1);
mean_arima_percent = zeros(iterations, 1);
mean_knn_rmse = zeros(iterations, 1);
mean_knn_percent = zeros(iterations, 1);

mean_elm_psa_relmae = zeros(iterations, 1);
mean_elm_relmae = zeros(iterations, 1);
mean_knn_relmae = zeros(iterations, 1);
mean_arima_relmae = zeros(iterations, 1);


for k = 1 : iterations
    Train = size(train_raw_data, 1);
    
    % Run normalized data
    train_data = normalized_data(1:Train, :);

    % Run PSA
    tic;
    [clusters_indexes, rep_members] = psa(train_data, nclusters);

    % Generate Training data sets from each cluster
    for i = 1 : nclusters
        cluster = train_data(clusters_indexes == i,:);
        input_data = generateInputData(cluster, window_size);
        elm_model(i) = trainELM(input_data, hidden_layer, 'sig');
    end
    train_time = toc;
    
    disp('PSA + ELM - Train Time');
    disp(train_time);
    
    % Generate Test data set
    test_data = normalized_data(Train+1:end, :);
    test_season_comp = smooth_data(Train+1:end, :);
    test_original_data = raw_data(Train+1:end, :);

    ndaily_samples = size(test_data, 2);
    test_input_data = generateInputData(test_data, window_size);
    test_season_comp_input_data = generateInputData(test_season_comp, window_size);
    test_original_input_data = generateInputData(test_original_data, window_size);
    
    ntestSamples = size(test_input_data.start_index, 1);
    dist_clusters = zeros(nclusters,1);
    accuracyRMSE = zeros(ntestSamples,1);
    accuracyPercent = zeros(ntestSamples,1);
    accuracyRelMAE = zeros(ntestSamples,1);
    persistenceForecast = zeros(ntestSamples,1);
    methodForecast = zeros(ntestSamples,1);
    
    % Test Persistence
    tic;
    for t = 1 : ntestSamples
        
        It_in = test_original_input_data.input(t, end);
        Ics_in = test_season_comp_input_data.input(t, end);

        It_out = test_original_input_data.output(t);
        Ics_out = test_season_comp_input_data.output(t);
       
        if Ics_in == 0
            restored_forecast_value = 0;
        else
            restored_forecast_value = Ics_out * (It_in / Ics_in);
        end
        
        persistenceForecast(t) = restored_forecast_value;
        accuracyRMSE(t) = sqrt(mse(It_out - restored_forecast_value));
        methodForecast(t) = restored_forecast_value;
        accuracyPercent(t) = 100 * abs(It_out - restored_forecast_value) / (max_data - min_data);
    end
    accuracynRMSE = sqrt(sum(((methodForecast - test_original_input_data.output).^2) /(sum(test_original_input_data.output .^2))));
    test_time = toc;
    
    disp('Persistence - Accuracy mean RMSE:');
    disp(mean(accuracyRMSE));
    disp('Persistence - Accuracy mean Percent:');
    disp(accuracynRMSE);
    

    tic;
    for t = 1 : ntestSamples
        start_index = test_input_data.start_index(t);

        % Decide which ELM model will be used for forecasting
        compare_index = start_index : start_index + window_size - 1;
        compare_index = compare_index - 1;
        compare_index = mod(compare_index, ndaily_samples);  
        compare_index = compare_index + 1;

        for i = 1 : nclusters
            compare_period = train_data(rep_members(i),compare_index);
            dist_clusters(i) = sqrt(sum((test_input_data.input(t,:) - compare_period) .^ 2));
        end

        [val, ind] = min(dist_clusters);

        % Test ELMs using each data set
        best_model = elm_model(ind);
        result = testELM(test_input_data, t, elm_model);
        
        denorm_forecast =  (((result.forecast + 1 )*(max_data - min_data)) + 2 * min_data) / 2;
        original_output = test_original_input_data.output(t);
        season_component = test_season_comp_input_data.output(t);
        restored_forecast_value = denorm_forecast + season_component;
       
        accuracyRMSE(t) = sqrt(mse(original_output - restored_forecast_value));
        accuracyRelMAE(t) = abs((restored_forecast_value -  persistenceForecast(t))/ (persistenceForecast(t) + eps));
        methodForecast(t) = restored_forecast_value;
    end
    accuracynRMSE = sqrt(sum(((methodForecast - test_original_input_data.output).^2) /(sum(test_original_input_data.output .^2))));
    test_time = toc;
    
    disp('PSA + ELM - Avg Test Time');
    disp(test_time / ntestSamples);


    disp('PSA + ELM - Accuracy mean RMSE:');
    disp(mean(accuracyRMSE));
    disp('PSA + ELM - Accuracy mean Percent:');
    disp(accuracynRMSE);
    disp('PSA + ELM - Relative MAE:');
    disp(mean(accuracyRelMAE));

    mean_elm_psa_rmse(k) = mean(accuracyRMSE);
    mean_elm_psa_percent(k) = accuracynRMSE;
    mean_elm_psa_relmae(k) = mean(accuracyRelMAE);


    % Run entire train data
    tic;
    entire_train_data = generateInputData(train_data, window_size);
    
    
    % Train ELM
    entire_elm_model = trainELM(entire_train_data, 4, 'sig');

    train_time = toc;
    
    disp('ELM - Train Time');
    disp(train_time);

    % ELM Test
    tic;
    for t = 1 : ntestSamples
        result = testELM(test_input_data, t, entire_elm_model);
        
        denorm_forecast =  (((result.forecast + 1 )*(max_data - min_data)) + 2 * min_data) / 2;
        original_output = test_original_input_data.output(t);
        season_component = test_season_comp_input_data.output(t);
        restored_forecast_value = denorm_forecast + season_component;
       
        accuracyRMSE(t) = sqrt(mse(original_output - restored_forecast_value));
        accuracyRelMAE(t) = abs((restored_forecast_value -  persistenceForecast(t))/ (persistenceForecast(t) + eps));
        methodForecast(t) = restored_forecast_value;
    end
    accuracynRMSE = sqrt(sum(((methodForecast - test_original_input_data.output).^2) /(sum(test_original_input_data.output .^2))));

    test_time = toc;
    
    disp('ELM - Avg Test Time');
    disp(test_time / ntestSamples);

    disp('ELM - Accuracy mean RMSE:');
    disp(mean(accuracyRMSE));
    disp('ELM - Accuracy mean Percent:');
    disp(accuracynRMSE);
    disp('ELM - Relative MAE:');
    disp(mean(accuracyRelMAE));
    mean_elm_relmae(k) = mean(accuracyRelMAE);

    % kNN Test
    accuracyRMSEKNN = zeros(ntestSamples,1);
    accuracyPercentKNN = zeros(ntestSamples,1);

    tic;
    for t = 1 : ntestSamples
        result = testKNN(entire_train_data, test_input_data, t);

        denorm_forecast =  (((result.forecast + 1 )*(max_data - min_data)) + 2 * min_data) / 2;
        original_output = test_original_input_data.output(t);
        season_component = test_season_comp_input_data.output(t);
        restored_forecast_value = denorm_forecast + season_component;
       
        accuracyRMSEKNN(t) = sqrt(mse(original_output - restored_forecast_value));
        accuracyPercentKNN(t) = 100 * sqrt((original_output - restored_forecast_value)^2 / (original_output^2 + eps));
        accuracyRelMAE(t) = abs((restored_forecast_value -  persistenceForecast(t))/ (persistenceForecast(t) + eps));
        methodForecast(t) = restored_forecast_value;
        
    end
    accuracynRMSE = sqrt(sum(((methodForecast - test_original_input_data.output).^2) /(sum(test_original_input_data.output .^2))));
    test_time = toc;
    
    disp('kNN - Avg Test Time');
    disp(test_time / ntestSamples);
    
    disp('kNN - Accuracy mean RMSE:');
    disp(mean(accuracyRMSEKNN));
    disp('kNN - Accuracy mean Percent:');
    disp(accuracynRMSE);
    disp('kNN - Relative MAE:');
    disp(mean(accuracyRelMAE));
    mean_knn_relmae(k) = mean(accuracyRelMAE);

    % ARIMA Estimate
    accuracyRMSEARIMA = zeros(ntestSamples,1);
    accuracyPercentARIMA = zeros(ntestSamples,1);
    model = arima(8,1,1);
    tic;
%     model = arima('AR',1,'MA',1,'Constant',0,'D',1,'Seasonality',48);
    fit = estimate(model,reshape(entire_train_data.input', 1, [])','Display','off');
    arima_train_time = toc;
    
    
    % ARIMA Test
    arima_test_input_data = generateInputData(test_data, 49);
    arima_test_season_comp_input_data = generateInputData(test_season_comp, 49);
    arima_test_original_input_data = generateInputData(test_original_data, 49);

    narimasamples = size(arima_test_input_data.start_index, 1);
    arimaMethodForecast = zeros(narimasamples,1);
    
    tic;
    for t = 1 : narimasamples
        
        % Persistence for ARIMA
        It_in = arima_test_original_input_data.input(t, end);
        Ics_in = arima_test_season_comp_input_data.input(t, end);

        It_out = arima_test_original_input_data.output(t);
        Ics_out = arima_test_season_comp_input_data.output(t);
       
        if Ics_in == 0
            persistence_forecast_value = 0;
        else
            persistence_forecast_value = Ics_out * (It_in / Ics_in);
        end

        % Test ARIMA
        result = testARIMA(fit, arima_test_input_data, t);

        denorm_forecast =  (((result.forecast + 1 )*(max_data - min_data)) + 2 * min_data) / 2;
        original_output = arima_test_original_input_data.output(t);
        season_component = arima_test_season_comp_input_data.output(t);
        restored_forecast_value = denorm_forecast + season_component;
       
        accuracyRMSEARIMA(t) = sqrt(mse(original_output - restored_forecast_value));
        accuracyPercentARIMA(t) = 100 * sqrt((original_output - restored_forecast_value)^2 / (original_output^2 + eps));
        accuracyRelMAE(t) = abs((restored_forecast_value -  persistence_forecast_value)/ (persistence_forecast_value + eps));
        arimaMethodForecast(t) = restored_forecast_value;

    end
    accuracynRMSE = sqrt(sum(((arimaMethodForecast - arima_test_original_input_data.output).^2) /(sum(arima_test_original_input_data.output .^2))));
    test_time = toc;

    disp('ARIMA - Train Time');
    disp(arima_train_time);
    
    disp('ARIMA - Avg Test Time');
    disp(test_time / narimasamples);


    disp('ARIMA - Accuracy mean RMSE:');
    disp(mean(accuracyRMSEARIMA));
    disp('ARIMA - Accuracy mean Percent:');
    disp(accuracynRMSE);
    disp('ARIMA - Relative MAE:');
    disp(mean(accuracyRelMAE));
    mean_arima_relmae(k) = mean(accuracyRelMAE);

    
    mean_elm_rmse(k) = mean(accuracyRMSE);
    mean_elm_percent(k) = mean(accuracyPercent);

    mean_knn_rmse(k) = mean(accuracyRMSEKNN);
    mean_knn_percent(k) = mean(accuracyPercentKNN);

    mean_arima_rmse(k) = mean(accuracyRMSEARIMA);
    mean_arima_percent(k) = mean(accuracyPercentARIMA);
end


disp('PSA + ELM - Total Accuracy RMSE mean :');
disp(mean(mean_elm_psa_rmse));
disp('PSA + ELM - Total Accuracy Percent mean :');
disp(mean(mean_elm_psa_percent));

disp('ELM - Total Accuracy RMSE mean :');
disp(mean(mean_elm_rmse));
disp('ELM - Total Accuracy Percent mean :');
disp(mean(mean_elm_percent));

disp('ARIMA - Total Accuracy RMSE mean :');
disp(mean(mean_arima_rmse));
disp('ARIMA - Total Accuracy Percent mean :');
disp(mean(mean_arima_percent));

disp('kNN - Total Accuracy RMSE mean :');
disp(mean(mean_knn_rmse));
disp('kNN - Total Accuracy Percent mean :');
disp(mean(mean_knn_percent));
