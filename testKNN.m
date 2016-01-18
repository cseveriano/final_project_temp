function [result] = testKNN(TrainData, TestData, index)

Test.out = TestData.output(index)';
Test.in = TestData.input(index,:)';

[distances] = calculateDistances(Test.in, TrainData);


[values, indexes] = sort(distances);

% If more than one sample have minimal distance, an average value is taken
min_dist_indexes = indexes(values == values(1));

if length(min_dist_indexes) == 1
    result.forecast = TrainData.output(min_dist_indexes);
else
    result.forecast = mean(TrainData.output(min_dist_indexes));
end

result.TestingAccuracyRMSE=sqrt(mse(Test.out - result.forecast));            %   Calculate testing accuracy (RMSE) for regression case
result.TestingAccuracyPercent=percent(Test.out - result.forecast);

end


function [distances] = calculateDistances(TestInput, TrainData)
    nTrain = size(TrainData.output);
    distances = zeros(nTrain);
    
    for  i = 1 : nTrain
        trainIn = TrainData.input(i,:)';
        distances(i) = sqrt(sum((TestInput - trainIn) .^ 2));
    end
end