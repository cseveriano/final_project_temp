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

% Split kfold
Indices = crossvalind('Kfold', size(normalized_data,1), kfold);
mean_elm_psa_rmse = zeros(kfold, 1);
mean_elm_psa_percent = zeros(kfold, 1);
mean_elm_rmse = zeros(kfold, 1);
mean_elm_percent = zeros(kfold, 1);

for k = 1 : kfold
    Train = Indices ~= k;
    Test = Indices == k;
    
    % Run normalized data
    train_data = normalized_data(Train, :);

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
    test_data = normalized_data(Test, :);
    ndaily_samples = size(test_data, 2);
    test_input_data = generateInputData(test_data, window_size);
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
            dist_clusters(i) = norm(test_input_data.input(t,:) - compare_period);
        end

        [val, ind] = min(dist_clusters);

        % Test ELMs using each data set
        best_model = elm_model(ind);
        result = testELM(test_input_data, t, elm_model);
        accuracyRMSE(t) = result.TestingAccuracyRMSE;
        accuracyPercent(t) = result.TestingAccuracyPercent;
    end
    test_time = toc;
    
    disp('PSA + ELM - Avg Test Time');
    disp(test_time / ntestSamples);


    disp('PSA + ELM - Accuracy mean RMSE:');
    disp(mean(accuracyRMSE));
    disp('PSA + ELM - Accuracy mean Percent:');
    disp(mean(accuracyPercent));

    mean_elm_psa_rmse(k) = mean(accuracyRMSE);
    mean_elm_psa_percent(k) = mean(accuracyPercent);


    % Run entire train data
    tic;
    entire_train_data = generateInputData(train_data, window_size);
    entire_elm_model = trainELM(entire_train_data, 4, 'sig');

    train_time = toc;
    
    disp('ELM - Train Time');
    disp(train_time);

    tic;
    for t = 1 : ntestSamples
        result = testELM(test_input_data, t, entire_elm_model);
        accuracyRMSE(t) = result.TestingAccuracyRMSE;
        accuracyPercent(t) = result.TestingAccuracyPercent;
    end
    test_time = toc;
    
    disp('ELM - Avg Test Time');
    disp(test_time / ntestSamples);

    disp('ELM - Accuracy mean RMSE:');
    disp(mean(accuracyRMSE));
    disp('ELM - Accuracy mean Percent:');
    disp(mean(accuracyPercent));

    mean_elm_rmse(k) = mean(accuracyRMSE);
    mean_elm_percent(k) = mean(accuracyPercent);
end


disp('PSA + ELM - Total Accuracy RMSE mean :');
disp(mean(mean_elm_psa_rmse));
disp('PSA + ELM - Total Accuracy Percent mean :');
disp(mean(mean_elm_psa_percent));

disp('ELM - Total Accuracy RMSE mean :');
disp(mean(mean_elm_rmse));
disp('ELM - Total Accuracy Percent mean :');
disp(mean(mean_elm_percent));
