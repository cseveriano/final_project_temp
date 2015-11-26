function [result] = testELM(TestData, model)

%%%%%%%%%%% Load test dataset
TV.T = TestData.output';
TV.P = TestData.input';
NumberofTestingData=size(TV.P,2);

%%%%%%%%%%% Calculate the output of testing input
start_time_test=cputime;
tempH_test=model.InputWeight*TV.P;
clear TV.P;             %   Release input of testing data             
ind=ones(1,NumberofTestingData);
BiasMatrix=model.BiasofHiddenNeurons(:,ind);              %   Extend the bias matrix BiasofHiddenNeurons to match the demention of H
tempH_test=tempH_test + BiasMatrix;
switch lower(model.ActivationFunction)
    case {'sig','sigmoid'}
        %%%%%%%% Sigmoid 
        H_test = 1 ./ (1 + exp(-tempH_test));
    case {'sin','sine'}
        %%%%%%%% Sine
        H_test = sin(tempH_test);        
    case {'hardlim'}
        %%%%%%%% Hard Limit
        H_test = hardlim(tempH_test);        
    case {'tribas'}
        %%%%%%%% Triangular basis function
        H_test = tribas(tempH_test);        
    case {'radbas'}
        %%%%%%%% Radial basis function
        H_test = radbas(tempH_test);        
        %%%%%%%% More activation functions can be added here        
end
TY=(H_test' * model.OutputWeight)';                       %   TY: the actual output of the testing data
end_time_test=cputime;
result.TestingTime=end_time_test-start_time_test;           %   Calculate CPU time (seconds) spent by ELM predicting the whole testing data

result.TestingAccuracy=sqrt(mse(TV.T - TY));            %   Calculate testing accuracy (RMSE) for regression case

end