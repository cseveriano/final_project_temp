function [training_data] = generateTrainingData(data, window_size)
    nsamples_daily = size(window_size,2);
    input = [];
    output = [];
    start_index = [];
    
    seq_data = reshape(data',numel(data),1);
    
    for i = 1 : numel(seq_data) - window_size - 1
        input = [input; seq_data(i:i+window_size-1)'];
        output = [output; seq_data(i+window_size)];
        start_index = [start_index; mod(i,nsamples_daily)];
    end
    
    training_data.input = input;
    training_data.output = output;
    training_data.start_index = start_index;
end