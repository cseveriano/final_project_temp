clear all;

dirPath =  'C:\Users\Carlos\Documents\Projetos Machine Learning\Multiagent\Data\Input\*\*\';
dirPath2 =  'C:\Users\Carlos\Documents\Projetos Machine Learning\ANN-CV\CODES\Git\AN-CV\Kyoto\Data\INPUTS\Delta\2013-04\2013-04[AVG-30]\';
file_pattern = '*[AVG-30].txt';
window_size = 8;
nclusters = 6;

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

% Split train and test data
[Train, Test] = crossvalind('HoldOut', size(normalized_data,1), 0.2);

% Run PSA
train_data = normalized_data(Train, :);
[clusters_indexes, rep_members] = psa(train_data, nclusters);

% Generate Training data sets from each cluster
for i = 1 : nclusters
    cluster = train_data(clusters_indexes == i,:);
    input_data = generateInputData(cluster, window_size);
    elm_model(i) = trainELM(input_data, 4, 'sig');
end


% Generate Test data set
test_data = normalized_data(Test, :);
ndaily_samples = size(test_data, 2);
test_input_data = generateInputData(test_data, window_size);
ntestSamples = size(test_input_data.start_index, 1);
dist_clusters = zeros(nclusters,1);
accuracy = zeros(ntestSamples,1);

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
    accuracy(t) = result.TestingAccuracy;
end


disp('PSA + ELM - Accuracy mean :');
disp(mean(accuracy));


% Run entire train data
entire_train_data = generateInputData(train_data, window_size);
entire_elm_model = trainELM(entire_train_data, 4, 'sig');

for t = 1 : ntestSamples
    result = testELM(test_input_data, t, entire_elm_model);
    accuracy(t) = result.TestingAccuracy;
end

disp('ELM - Accuracy mean :');
disp(mean(accuracy));


