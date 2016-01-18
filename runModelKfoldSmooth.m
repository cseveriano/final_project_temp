clear all;

dirPath =  'C:\Users\Carlos\Documents\Projetos Machine Learning\Multiagent\Data\Input\*\*\';

file_pattern = '*[AVG-30].txt';
window_size = 8;
nclusters = 8;
hidden_layer = 6;
kfold = 5;

% Read data files
files = rdir(strcat(dirPath, file_pattern));
raw_data = [];

for i = 1 : size(files)
    x = importdata(files(i).name,'\t', 1);
    raw_data = [raw_data; x.data(:,1)'];
    clear x;
end

% Normalize data
normalized_data = -1 + 2.*(raw_data - min(min(raw_data)))./(max(max(raw_data)) - min(min(raw_data)));
clear raw_data;

season_component = reshape(smooth(normalized_data), [], size(normalized_data, 2));
deseason_data = normalized_data - season_component;

% Split kfold
Indices = crossvalind('Kfold', size(normalized_data,1), kfold);
mean_elm_psa_rmse = zeros(kfold, 1);
mean_elm_psa_percent = zeros(kfold, 1);
mean_elm_rmse = zeros(kfold, 1);
mean_elm_percent = zeros(kfold, 1);
mean_arima_rmse = zeros(kfold, 1);
mean_arima_percent = zeros(kfold, 1);
mean_knn_rmse = zeros(kfold, 1);
mean_knn_percent = zeros(kfold, 1);


for k = 1 : kfold
    Train = Indices ~= k;
    Test = Indices == k;
    
    % Run normalized data
    train_data = deseason_data(Train, :);

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
    test_data = deseason_data(Test, :);
    test_season_comp = season_component(Test, :);
    test_original_data = normalized_data(Test, :);

    ndaily_samples = size(test_data, 2);
    test_input_data = generateInputData(test_data, window_size);
    test_season_comp_input_data = generateInputData(test_season_comp, window_size);
    test_original_input_data = generateInputData(test_original_data, window_size);
    
    ntestSamples = size(test_input_data.start_index, 1);
    dist_clusters = zeros(nclusters,1);
    accuracyRMSE = zeros(ntestSamples,1);
    accuracyPercent = zeros(ntestSamples,1);

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
%             dist_clusters(i) =  norm(test_input_data.input(t,:) - compare_period);
            dist_clusters(i) = sqrt(sum((test_input_data.input(t,:) - compare_period) .^ 2));
        end

        [val, ind] = min(dist_clusters);

        % Test ELMs using each data set
        best_model = elm_model(ind);
        result = testELM(test_input_data, t, elm_model);
        
        original_output = test_original_input_data.output(t);
        season_component = test_season_comp_input_data.output(t);
        restored_forecast_value = result.forecast + season_component;
       
        accuracyRMSE(t) = sqrt(mse(original_output - restored_forecast_value));
        accuracyPercent(t) = percent(original_output - restored_forecast_value);
    end
    test_time = toc;
    
    disp('PSA + ELM - Avg Test Time');
    disp(test_time / ntestSamples);


    disp('PSA + ELM - Accuracy mean RMSE:');
    disp(mean(accuracyRMSE));
    disp('PSA + ELM - Accuracy mean Percent:');
    disp(mean(accuracyPercent));

    
% COMPARE WITH NORMAL MODEL (NO DESEASON)

%     % Run normalized data
%     train_data = normalized_data(Train, :);
% 
%     % Run PSA
%     tic;
%     [clusters_indexes, rep_members] = psa(train_data, nclusters);
% 
%     % Generate Training data sets from each cluster
%     for i = 1 : nclusters
%         cluster = train_data(clusters_indexes == i,:);
%         input_data = generateInputData(cluster, window_size);
%         elm_model(i) = trainELM(input_data, hidden_layer, 'sig');
%     end
%     train_time = toc;
%     
%     disp('PSA + ELM - Train Time');
%     disp(train_time);
%     
%     % Generate Test data set
%     test_data = normalized_data(Test, :);
% 
%     ndaily_samples = size(test_data, 2);
%     test_input_data = generateInputData(test_data, window_size);
%     
%     ntestSamples = size(test_input_data.start_index, 1);
%     dist_clusters = zeros(nclusters,1);
%     accuracyRMSE = zeros(ntestSamples,1);
%     accuracyPercent = zeros(ntestSamples,1);
% 
%     tic;
%     for t = 1 : ntestSamples
%         start_index = test_input_data.start_index(t);
% 
%         % Decide which ELM model will be used for forecasting
%         compare_index = start_index : start_index + window_size - 1;
%         compare_index = compare_index - 1;
%         compare_index = mod(compare_index, ndaily_samples);  
%         compare_index = compare_index + 1;
% 
%         for i = 1 : nclusters
%             compare_period = train_data(rep_members(i),compare_index);
% %             dist_clusters(i) =  norm(test_input_data.input(t,:) - compare_period);
%             dist_clusters(i) = sqrt(sum((test_input_data.input(t,:) - compare_period) .^ 2));
%         end
% 
%         [val, ind] = min(dist_clusters);
% 
%         % Test ELMs using each data set
%         best_model = elm_model(ind);
%         result = testELM(test_input_data, t, elm_model);
%         
%         original_output = test_input_data.output(t);
%         restored_forecast_value = result.forecast;
%        
%         accuracyRMSE(t) = sqrt(mse(original_output - restored_forecast_value));
%         accuracyPercent(t) = percent(original_output - restored_forecast_value);
%     end
%     test_time = toc;
%     
%     disp('NORMAL PSA + ELM - Avg Test Time');
%     disp(test_time / ntestSamples);
% 
% 
%     disp('NORMAL PSA + ELM - Accuracy mean RMSE:');
%     disp(mean(accuracyRMSE));
%     disp('NORMAL PSA + ELM - Accuracy mean Percent:');
%     disp(mean(accuracyPercent));

% --------
    mean_elm_psa_rmse(k) = mean(accuracyRMSE);
    mean_elm_psa_percent(k) = mean(accuracyPercent);


    % Run entire train data
    tic;
    entire_train_data = generateInputData(train_data, window_size);
    entire_elm_model = trainELM(entire_train_data, 4, 'sig');

    train_time = toc;
    
    disp('ELM - Train Time');
    disp(train_time);

    % ELM Test
    tic;
    for t = 1 : ntestSamples
        result = testELM(test_input_data, t, entire_elm_model);
        accuracyRMSE(t) = result.TestingAccuracyRMSE;
        accuracyPercent(t) = result.TestingAccuracyPercent;
    end
    test_time = toc;
    
    disp('ELM - Avg Test Time');
    disp(test_time / ntestSamples);
    
    % kNN Test
%     accuracyRMSEKNN = zeros(ntestSamples,1);
%     accuracyPercentKNN = zeros(ntestSamples,1);
% 
%     tic;
%     for t = 1 : ntestSamples
%         result = testKNN(entire_train_data, test_input_data, t);
%         accuracyRMSEKNN(t) = result.TestingAccuracyRMSE;
%         accuracyPercentKNN(t) = result.TestingAccuracyPercent;
%     end
%     test_time = toc;
%     
%     disp('kNN - Avg Test Time');
%     disp(test_time / ntestSamples);
    

    % ARIMA Test
    accuracyRMSEARIMA = zeros(ntestSamples,1);
    accuracyPercentARIMA = zeros(ntestSamples,1);
%     model = arima(0,1,1);
    model = arima('Constant',0,'D',1,'Seasonality',48);
    
    fit = estimate(model,reshape(entire_train_data.input', 1, [])','Display','off');
    arima_test_input_data = generateInputData(test_data, 49);

    tic;
    for t = 1 : size(arima_test_input_data.start_index, 1)
        result = testARIMA(fit, arima_test_input_data, t);
        accuracyRMSEARIMA(t) = result.TestingAccuracyRMSE;
        accuracyPercentARIMA(t) = result.TestingAccuracyPercent;
    end
    test_time = toc;

    disp('ARIMA - Avg Test Time');
    disp(test_time / ntestSamples);

    disp('ELM - Accuracy mean RMSE:');
    disp(mean(accuracyRMSE));
    disp('ELM - Accuracy mean Percent:');
    disp(mean(accuracyPercent));

    disp('kNN - Accuracy mean RMSE:');
    disp(mean(accuracyRMSEKNN));
    disp('kNN - Accuracy mean Percent:');
    disp(mean(accuracyPercentKNN));

    disp('ARIMA - Accuracy mean RMSE:');
    disp(mean(accuracyRMSEARIMA));
    disp('ARIMA - Accuracy mean Percent:');
    disp(mean(accuracyPercentARIMA));

    mean_elm_rmse(k) = mean(accuracyRMSE);
    mean_elm_percent(k) = mean(accuracyPercent);

    mean_knn_rmse(k) = mean(accuracyRMSEKNN);
    mean_knn_percent(k) = mean(accuracyPercentKNN);

%     mean_arima_rmse(k) = mean(accuracyRMSEARIMA);
%     mean_arima_percent(k) = mean(accuracyPercentARIMA);
end


disp('PSA + ELM - Total Accuracy RMSE mean :');
disp(mean(mean_elm_psa_rmse));
disp('PSA + ELM - Total Accuracy Percent mean :');
disp(mean(mean_elm_psa_percent));

disp('ELM - Total Accuracy RMSE mean :');
disp(mean(mean_elm_rmse));
disp('ELM - Total Accuracy Percent mean :');
disp(mean(mean_elm_percent));

% disp('ARIMA - Total Accuracy RMSE mean :');
% disp(mean(mean_arima_rmse));
% disp('ARIMA - Total Accuracy Percent mean :');
% disp(mean(mean_arima_percent));

disp('kNN - Total Accuracy RMSE mean :');
disp(mean(mean_knn_rmse));
disp('kNN - Total Accuracy Percent mean :');
disp(mean(mean_knn_percent));
