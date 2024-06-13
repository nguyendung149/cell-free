clear all,close all,clc;

% Loading input to the NN
dataDNN = load('D:\DuAn\code\data_training.mat');
mu_RZF_sumSE_DNN = dataDNN.mu_RZF_sumSE_DNN;
betas_DNN = dataDNN.betas_DNN;

% Maximum downlink transmit power per BS (mW)
Pmax = 1000;
K = size(betas_DNN,1);
L = size(betas_DNN,2);
NoOfSetups = size(betas_DNN,3);
% Make sure the sum over the K UEs gives Pmax for each AP in each setup- (might not be necessary)
for n = 1:NoOfSetups
    mu_RZF_sumSE_DNN(:,:,n) = mu_RZF_sumSE_DNN(:,:,n) .* sqrt(Pmax/(max(sum(mu_RZF_sumSE_DNN(:,:,n).^2,1))));
end

cluster_size = 3;
numOfFeature = cluster_size*K;
modelArray = [];
%%%%%%%%%%%%%%%%%%%%%%%%%%%    MODEL FFNN   ##########################
for l = 1:cluster_size:L
     layers = [
        featureInputLayer(numOfFeature,Name='input',Normalization='rescale-symmetric')
        fullyConnectedLayer(128,WeightsInitializer="narrow-normal",BiasInitializer="narrow-normal",Name='Layer 1')
        batchNormalizationLayer
        leakyReluLayer
        fullyConnectedLayer(512,WeightsInitializer="narrow-normal",BiasInitializer="narrow-normal",Name='Layer 2')
        batchNormalizationLayer
        eluLayer
        fullyConnectedLayer(256,WeightsInitializer="narrow-normal",BiasInitializer="narrow-normal",Name="Layer 3")
        batchNormalizationLayer
        tanhLayer
        fullyConnectedLayer(128,WeightsInitializer="narrow-normal",BiasInitializer="narrow-normal",Name="Layer 4")
        batchNormalizationLayer
        tanhLayer
        fullyConnectedLayer(cluster_size*(K+1) ,WeightsInitializer="narrow-normal",BiasInitializer="narrow-normal",Name='Layer 5')
        reluLayer
];
    NoOfSetups = size(betas_DNN,3);
    % Preparing inputs for NN
    % beta vector preparation (removing outliers and scaling)
    if l == 16
        l = 14;
    end
    betas = zeros(NoOfSetups,cluster_size*K);
    for c = 1:cluster_size
        betas(:,(c-1)*K+1:c*K) = reshape(betas_DNN(:,l+c-1,:),K,[]).';
    end
    betas = 10*log10(betas*1000);
    big_values = [];
    for i = 1:NoOfSetups
        if any(betas(i,:) > 37)
            big_values = [big_values i];
        end
    end
    betas(big_values,:) = [];
    NoOfSetups =size(betas,1);

    betas = robustScaler(betas,0,1);
    DNNinput = betas;
    x_train = DNNinput(1:NoOfSetups-100,:);

    mu = zeros(NoOfSetups,cluster_size*K);
    temp = zeros(NoOfSetups-100,cluster_size);
    for c = 1:cluster_size
        store = reshape(mu_RZF_sumSE_DNN(:,l+c-1,:),K,[]).';
        store(big_values,:) = [];
        mu(:,(c-1)*K+1:c*K) = abs(store);
        temp(:,c) = sqrt(reshape(sum((mu(1:NoOfSetups - 100,(c-1)*K+1:c*K).').^2),NoOfSetups-100,1)/Pmax);
    end
    y_train = [mu(1:NoOfSetups-100,:) temp];

    small_values = [];
    small_val = 5/sqrt(Pmax);
    for i = 1:(NoOfSetups-100)
        counter = 0;
        for j = 1:cluster_size
            if y_train(i,cluster_size*K+j)<small_val
                counter = counter + 1;
            end
        end
        if counter == cluster_size
            small_values = [small_values i];
        end
    end
    y_train(small_values,:) = [];
    NoOfSetups = size(y_train,1);

    % Normalization with sqrt(Pmax) must be done separately for each AP as follows
    for c = 1:cluster_size
        y_train(:,(c-1)*K+1:c*K) = sqrt(K) * normalize(y_train(:,(c-1)*K+1:c*K).',"norm",2).';
    end


    x_train(small_values,:) = [];

    % Validation data
    % Cross varidation 
    cv = cvpartition(size(x_train,1),'HoldOut',0.1);
    idx = cv.test;
    % Separate to training and test data
    dataTrain_X = x_train(~idx,:);
    dataValidation_X  = x_train(idx,:);
    
    dataTrain_y = y_train(~idx,:);
    dataValidation_y = y_train(idx,:);

    x_test = DNNinput(NoOfSetups-99:NoOfSetups,:);
    % Assign y_test based on the model choice above
    y_test = mu(NoOfSetups-99:NoOfSetups,:);
    for c = 1:cluster_size
        y_test(:,(c-1)*K+1:c*K) = sqrt(K)*normalize(y_test(:,(c-1)*K+1:c*K).',"norm",2).';
    end
    
     options = trainingOptions("adam", ...
    "LearnRateDropFactor",0.2,...
    "LearnRateDropPeriod",5,...
    "SquaredGradientDecayFactor",0.999,...
    "MaxEpochs",50, ...,
    "ValidationPatience",20,...
    "Plots","training-progress", ...
    "Metrics","rmse",...
    "Verbose",true, ...
    "Epsilon",1e-7, ...
    "InitialLearnRate",0.001, ...
    "MiniBatchSize",128, ...
    "LearnRateSchedule","piecewise", ...
    "OutputNetwork","best-validation-loss", ...
    "Shuffle","every-epoch",...
    "ValidationData",{dataValidation_X,dataValidation_y});
    
     %% Training
     netTrained = trainnet(dataTrain_X,dataTrain_y,layers,"mse",options);
     y_predictions = netTrained.predict(x_test);
     test_mse = mean(reshape((y_test - y_predictions(:,1:cluster_size*K)).^2,1,[]));
     test_mseAP4 = mean(reshape((y_test(:,1:20) - y_predictions(:,1:20)).^2,1,[]));
     
     fprintf("Test MSE: %f \t",test_mse);
     fprintf("Test MSEAP4: %f",test_mseAP4);
     modelArray = [modelArray netTrained];

end

