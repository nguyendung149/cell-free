clear all,close all,clc;

%% Load data
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

%%%%%%%%%%%%%%%%%%%%%%%%%%%    MODEL FFNN   ##########################
numOfFeature = 20;
modelArray = [];
for l = 1:1
    % Preparing inputs for NN
    % beta vector preparation(removing outliers and scaling)
    betas = reshape(betas_DNN(:,l,:),K,[]).'; % or use a function of betas
    betas = 10*log10(betas*1000); % dB scale
    big_values = [];
    for i = 1:NoOfSetups
        if any(betas(i,:) > 34)
            big_values = [big_values i];
        end
    end
    
    betas(big_values,:) = [];
    NoOfSetups = size(betas,1);


    betas = 10.^(betas/10); %  # changing back to linear scale


    % The betas are changed to mus with fractional power allocatrion (gives better scaling)
    v = 0.6; % Fractional power allocation factor
    betas = sqrt(Pmax) * ((betas.^v)./reshape(sum(betas.^v,2),NoOfSetups,1));
    betas = 10*log10(betas); %db scale

    % betas = normalize(betas,"scale","iqr","center","median");
    betas = robustScaler(betas,0,1);

    DNNinput = betas;
    x_train = DNNinput(1:NoOfSetups - 100,:);

    % mu preparation
    mu = reshape(abs(mu_RZF_sumSE_DNN(:,l,:)),K,[]).';
    mu(big_values,:) = [];
    temp = sqrt(reshape(sum((mu(1:NoOfSetups - 100,:).').^2),NoOfSetups-100,1)/Pmax);
    y_train = [mu(1:NoOfSetups-100,:) temp];
    small_values = [];
    small_val = 5 / sqrt(Pmax);
    for i = 1:NoOfSetups-100
        if y_train(i,K+1) < small_val
            small_values = [small_values i ];
        end

    end
    y_train(small_values,:) = [];
    NoOfSetups = size(y_train,1);
    y_train(y_train < 0.001) = 0.001;
    y_train(:,1:K) = sqrt(K) * normalize(y_train(:,1:K).',"norm",2).';
    x_train(small_values,:) = [];
    
    % Validation data
    % Cross varidation (train: 70%, test: 10%)
    cv = cvpartition(size(x_train,1),'HoldOut',0.1);
    idx = cv.test;
    % Separate to training and test data
    dataTrain_X = x_train(~idx,:);
    dataValidation_X  = x_train(idx,:);
    
    dataTrain_y = y_train(~idx,:);
    dataValidation_y = y_train(idx,:);

    y_test = mu(NoOfSetups - 99 : NoOfSetups,:);
    y_test = sqrt(K) * normalize(y_test(:,1:K).',"norm",2).';
    x_test = DNNinput((NoOfSetups - 99):NoOfSetups,:);
    
%% Training
    options = trainingOptions("adam", ...
    "LearnRateDropFactor",0.2,...
    "LearnRateDropPeriod",5,...
    "SquaredGradientDecayFactor",0.999,...
    "MaxEpochs",30, ...,
    "ValidationPatience",50,...
    "Plots","training-progress", ...
    "Metrics","rmse",...
    "Verbose",true, ...
    "Epsilon",1e-7, ...
    "InitialLearnRate",0.001, ...
    "MiniBatchSize",128, ...
    "LearnRateSchedule","piecewise", ...
    "OutputNetwork","last-iteration", ...
    "Shuffle","every-epoch",...
    "ValidationData",{dataValidation_X,dataValidation_y});
    layers = [
        featureInputLayer(numOfFeature,Name='input',Normalization='rescale-symmetric')
        fullyConnectedLayer(32,WeightsInitializer="narrow-normal",BiasInitializer="narrow-normal",Name='Layer1')
        batchNormalizationLayer
        leakyReluLayer
        fullyConnectedLayer(64,WeightsInitializer="narrow-normal",BiasInitializer="narrow-normal",Name="Layer 3")
        batchNormalizationLayer
        tanhLayer
        fullyConnectedLayer(32,WeightsInitializer="narrow-normal",BiasInitializer="narrow-normal",Name="Layer 4")
        batchNormalizationLayer
        tanhLayer
        fullyConnectedLayer(K + 1 ,WeightsInitializer="narrow-normal",BiasInitializer="narrow-normal",Name='Layer 5')
        reluLayer
];
    netTrained = trainnet(dataTrain_X,dataTrain_y,layers,"mse",options);
    y_predictions = predict(netTrained,x_test);
    test_mse = mean((y_test - y_predictions(:,1:K)).^2);
    fprintf("Test MSE: %f",test_mse)
    modelArray = [modelArray netTrained];
end