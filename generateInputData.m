function [training_data] = generateInputData(data, window_size)
    nsamples_daily = size(data,2);
    input = [];
    output = [];
    start_index = [];
    
    seq_data = reshape(data',numel(data),1);
    
    for i = 1 : numel(seq_data) - window_size - 1
        input = [input; seq_data(i:i+window_size-1)'];
        output = [output; seq_data(i+window_size)];
        st = mod(i,nsamples_daily);
        
        if (st == 0)
            st = nsamples_daily;
        end
                       
        start_index = [start_index; st];
    end
    
    training_data.input = input;
    training_data.output = output;
    training_data.start_index = start_index;
end