clear all,close all,clc;

% Loading input to the NN
dataDNN = load('D:\DuAn\code\data_training.mat');
mu_RZF_PF_DNN = dataDNN.mu_RZF_PF_DNN;
betas_DNN = dataDNN.betas_DNN;

% Maximum downlink transmit power per BS (mW)
Pmax = 1000;
K = size(betas_DNN,1);
L = size(betas_DNN,2);
NoOfSetups = size(betas_DNN,3);
% Make sure the sum over the K UEs gives Pmax for each AP in each setup- (might not be necessary)
for n = 1:NoOfSetups
    mu_RZF_PF_DNN(:,:,n) = mu_RZF_PF_DNN(:,:,n) .* sqrt(Pmax/(max(sum(mu_RZF_PF_DNN(:,:,n).^2,1))));
end

cluster_size = 3;
numOfFeature = cluster_size*K;

%%%%%%%%%%%%%%%%%%%%%%%%%%%    MODEL FFNN   ##########################
for l = 1:cluster_size:L
     
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
        store = reshape(mu_RZF_PF_DNN(:,l+c-1,:),K,[]).';
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

    %% Training options
    minibatch_size = 128;
    Training_set_ratio = 0.95;
    numEpochs = 30;
    learnRate = 2e-3; % 1.2e-3
    %Dropperiod = [23, 29, 30, 39, 49];
    Dropperiod = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90];
    Droprate = 0.5;
    L2Regularization = 0.0000000001; % 0.0000000001
    Validation_frequency = 100;

    load_parameters = false;
    parameter_file = 'parameters_EPA';

    gpuDevice(1)
    disp(gpuDeviceTable);
    
    % Reshape dimension data
    dataTrain_X = dataTrain_X.';
    Training_X = reshape(dataTrain_X,size(dataTrain_X,1),1,1,[]);

    dataTrain_y = dataTrain_y.';
    Training_Y = reshape(dataTrain_y,size(dataTrain_y,1),1,1,[]);

    dataValidation_X = dataValidation_X.';
    Validation_X = reshape(dataValidation_X,size(dataValidation_X,1),1,1,[]);

    dataValidation_y = dataValidation_y.';
    Validation_Y = reshape(dataValidation_y,size(dataValidation_y,1),1,1,[]);

    x_test = x_test.';
    x_test = reshape(x_test,size(x_test,1),1,1,[]);

    y_test = y_test.';
    y_test = reshape(y_test,size(y_test,1),1,1,[]);
    
    X = arrayDatastore(reshape(Training_X, size(Training_X, 1), size(Training_X, 2), size(Training_X, 4)), 'IterationDimension', 3);
    Y = arrayDatastore(reshape(Training_Y, size(Training_Y, 1), size(Training_Y, 2), size(Training_Y, 4)), 'IterationDimension', 3);
    cdsTrain = combine(X, Y);

    mbqTrain = minibatchqueue(cdsTrain, 2,...
        'MiniBatchSize', minibatch_size,...
        'MiniBatchFcn', @preprocessMiniBatch,...
        'MiniBatchFormat', {'',''},...
        'PartialMiniBatch', 'discard');

    cdsValidation = combine(arrayDatastore(reshape(Validation_X, size(Training_X, 1), size(Training_X, 2), size(Validation_X, 4)), 'IterationDimension', 3), arrayDatastore(reshape(Validation_Y,...
        size(Validation_Y, 1), size(Validation_Y, 2),size(Validation_Y, 4)), 'IterationDimension', 3));

    mbqValidation = minibatchqueue(cdsValidation, 2,...
        'MiniBatchSize', minibatch_size,...
        'MiniBatchFcn', @preprocessMiniBatch,...
        'MiniBatchFormat', {'',''},...
        'PartialMiniBatch', 'discard');

    shuffle(mbqValidation);

    Feature_size = size(Training_X, 1);

    %% Initialize

    if load_parameters == true
    
        load(parameter_file);
    
    else
    
        parameters.Hyperparameters.NumHeads = 2;
        parameters.Hyperparameters.Encoder_num_layers = 1;
        parameters.Hyperparameters.Decoder_num_layers = 1;
    
        for i = 1 : parameters.Hyperparameters.Encoder_num_layers
        
            parameters.Weights.encoder_layer.("layer_"+i).ln_1_g_0 = initializeGlorot([Feature_size, 1], Feature_size, Feature_size); % dlarray(rand(Feature_size, 1) / 1e10);
            parameters.Weights.encoder_layer.("layer_"+i).ln_1_b_0 = dlarray(zeros(Feature_size, 1));

            parameters.Weights.encoder_layer.("layer_"+i).ln_2_g_0 = initializeGlorot([Feature_size, 1], Feature_size, Feature_size);
            parameters.Weights.encoder_layer.("layer_"+i).ln_2_b_0 = dlarray(zeros(Feature_size, 1));

            parameters.Weights.encoder_layer.("layer_"+i).attn_c_attn_w_0 = initializeGlorot([3 * Feature_size, Feature_size], prod([3 * Feature_size, Feature_size]), prod([Feature_size, Feature_size]));
            parameters.Weights.encoder_layer.("layer_"+i).attn_c_attn_b_0 = dlarray(zeros(3 * Feature_size, 1));

            parameters.Weights.encoder_layer.("layer_"+i).attn_c_proj_w_0 = initializeGlorot([Feature_size, Feature_size], prod([Feature_size, Feature_size]), prod([Feature_size, Feature_size]));
            parameters.Weights.encoder_layer.("layer_"+i).attn_c_proj_b_0 = dlarray(zeros(Feature_size, 1));
            parameters.Weights.encoder_layer.("layer_"+i).mlp_c_fc_w_0 = initializeGlorot([Feature_size, Feature_size], prod([Feature_size, Feature_size]), prod([Feature_size, Feature_size]));
            parameters.Weights.encoder_layer.("layer_"+i).mlp_c_fc_b_0 = dlarray(zeros(Feature_size, 1));
            parameters.Weights.encoder_layer.("layer_"+i).mlp_c_proj_w_0 = initializeGlorot([Feature_size, Feature_size], prod([Feature_size, Feature_size]), prod([Feature_size, Feature_size]));
            parameters.Weights.encoder_layer.("layer_"+i).mlp_c_proj_b_0 = dlarray(zeros(Feature_size, 1));
    
        end
    
        Parameter.parameters_residual_neural_network
    
    end

    %% Train the model using a custom training loop

    % For each epoch, shuffle the mini-batch queue and loop over mini-batches
    % of data. At the end of each iteration, update the training progress plot
    %
    % For each iteration:
    % * Read a mini-batch of data from the mini-batch queue. 
    % * Evaluate the model gradients and loss using the |dlfeval| and
    %   |modelGradients| functions
    % * Update the network parameters using the |adamupdate| function.
    % * Update the training plot

    % Initialize training progress plot
    figure
    lineLossTrain = animatedline("Color", [0.8500 0.3250 0.0980]);
    lineLossValidation = animatedline("Color", [0 0.4470 0.7410]);

    ylim([0 1]);
    xlabel("Iteration");
    ylabel("Loss");

    % Initialize parameters for the Adam optimizer
    trailingAvg = [];
    trailingAvgSq = [];

    iteration = 0;
    start = tic;

    % Loop over epochs
    for epoch = 1 : numEpochs
    
        % Shuffle data
        shuffle(mbqTrain);
    
        % Loop over mini-batches
        while hasdata(mbqTrain)
            iteration = iteration + 1;
        
            % Read mini-batch of data
            [Training_X_minibatch, Training_Y_minibatch] = next(mbqTrain);
        
            if hasdata(mbqValidation)
                [Xvalidation_minibatch, Yvalidation_minibatch] = next(mbqValidation);
            else
                reset(mbqValidation);
            end
        
            % Evaluate loss and gradients
            [loss, gradients] = dlfeval(@modelGradients, gpuArray(Training_X_minibatch), gpuArray(Training_Y_minibatch), parameters);
        
            % Update model parameters
            if ismember(epoch, Dropperiod)
                learnRate = learnRate * Droprate;
            end
        
            for j = 1 : parameters.Hyperparameters.Encoder_num_layers
            
                gradients.encoder_layer.("layer_"+j).ln_1_g_0 = dlupdate(@(g, w)  g + (w * L2Regularization), gradients.encoder_layer.("layer_"+j).ln_1_g_0, parameters.Weights.encoder_layer.("layer_"+j).ln_1_g_0); % (w * L2Regularization)
                gradients.encoder_layer.("layer_"+j).ln_2_g_0 = dlupdate(@(g, w)  g + (w * L2Regularization), gradients.encoder_layer.("layer_"+j).ln_2_g_0, parameters.Weights.encoder_layer.("layer_"+j).ln_2_g_0);
                gradients.encoder_layer.("layer_"+j).attn_c_attn_w_0 = dlupdate(@(g, w) g + (w * L2Regularization), gradients.encoder_layer.("layer_"+j).attn_c_attn_w_0, parameters.Weights.encoder_layer.("layer_"+j).attn_c_attn_w_0);
                gradients.encoder_layer.("layer_"+j).attn_c_proj_w_0 = dlupdate(@(g, w) g + (w * L2Regularization), gradients.encoder_layer.("layer_"+j).attn_c_proj_w_0, parameters.Weights.encoder_layer.("layer_"+j).attn_c_proj_w_0);
                gradients.encoder_layer.("layer_"+j).mlp_c_fc_w_0 = dlupdate(@(g, w) g + (w * L2Regularization), gradients.encoder_layer.("layer_"+j).mlp_c_fc_w_0, parameters.Weights.encoder_layer.("layer_"+j).mlp_c_fc_w_0);
                gradients.encoder_layer.("layer_"+j).mlp_c_proj_w_0 = dlupdate(@(g, w) g + (w * L2Regularization), gradients.encoder_layer.("layer_"+j).mlp_c_proj_w_0, parameters.Weights.encoder_layer.("layer_"+j).mlp_c_proj_w_0);
        
            end
        
            for j = 1 : parameters.Hyperparameters.Decoder_num_layers
            
                gradients.decoder_layer.("layer_"+j).ln_de_w1 = dlupdate(@(g, w)  g + (w * L2Regularization), gradients.decoder_layer.("layer_"+j).ln_de_w1, parameters.Weights.decoder_layer.("layer_"+j).ln_de_w1); % (w * L2Regularization)
                gradients.decoder_layer.("layer_"+j).ln_de_w2 = dlupdate(@(g, w)  g + (w * L2Regularization), gradients.decoder_layer.("layer_"+j).ln_de_w2, parameters.Weights.decoder_layer.("layer_"+j).ln_de_w2);
                gradients.decoder_layer.("layer_"+j).ln_de_w3 = dlupdate(@(g, w) g + (w * L2Regularization), gradients.decoder_layer.("layer_"+j).ln_de_w3, parameters.Weights.decoder_layer.("layer_"+j).ln_de_w3);
            
            end
        
            gradients.decoder_layer.("layer_" + parameters.Hyperparameters.Decoder_num_layers + 1).ln_de_w1 = dlupdate(@(g, w)  g + (w * L2Regularization), gradients.decoder_layer.("layer_" + parameters.Hyperparameters.Decoder_num_layers + 1).ln_de_w1, parameters.Weights.decoder_layer.("layer_" + parameters.Hyperparameters.Decoder_num_layers + 1).ln_de_w1);
            gradients.decoder_layer.("layer_" + parameters.Hyperparameters.Decoder_num_layers + 1).ln_de_w2 = dlupdate(@(g, w)  g + (w * L2Regularization), gradients.decoder_layer.("layer_" + parameters.Hyperparameters.Decoder_num_layers + 1).ln_de_w2, parameters.Weights.decoder_layer.("layer_" + parameters.Hyperparameters.Decoder_num_layers + 1).ln_de_w2);
        
            [parameters.Weights, trailingAvg, trailingAvgSq] = adamupdate(parameters.Weights, gradients, ...
                trailingAvg, trailingAvgSq, iteration,learnRate);
        
            % Update training plot
            loss = double(gather(extractdata(loss)));
            addpoints(lineLossTrain, iteration, loss);
        
            if iteration == 1 || mod(iteration, Validation_frequency) == 0
            
                % Validation set
                Prediction_validation = transformer.model(Xvalidation_minibatch, parameters);
                %loss_validation = Myloss(Yvalidation_minibatch, Prediction_validation) / 100;
                %loss_validation = huber(Yvalidation_minibatch, Prediction_validation, "DataFormat", "SSCB", 'TransitionPoint', 1);
                %loss_validation = mse(change_dimension(Yvalidation_minibatch), Prediction_validation, "DataFormat", "SSCB");
                loss_validation = Myloss1(Yvalidation_minibatch,Prediction_validation);
                loss_validation = double(gather(extractdata(loss_validation)));
                addpoints(lineLossValidation, iteration, loss_validation);
            
            end
        
            disp("loss = " + loss)
            disp("Validation loss = " + loss_validation)
        
            D = duration(0,0,toc(start),'Format','hh:mm:ss');
            title("Epoch: " + epoch + ", Elapsed: " + string(D))
            drawnow
        end
    end
    y_predictions = transformer.model(x_test, parameters);
    test_mse = mean(reshape((y_test - y_predictions(1:numOfFeature,:,:,:)).^2,1,[]));
    fprintf("Test MSE: %f",test_mse)
    save("RZF_PF_CANN_"+l+".mat","parameters");
end
%% Supporting Functions

function [loss, gradients] = modelGradients(X, Y, parameters)

    Prediction = transformer.model(X, parameters);
    %loss = huber(Y, Prediction, "DataFormat", "SSCB", 'TransitionPoint', 1); % , "DataFormat", "SCB" huber change_dimension(Y)
    %loss = mse(change_dimension(Y), Prediction, "DataFormat", "SSCB");
    %loss = Myloss(change_dimension(Y), Prediction); % L1
    loss = Myloss1(Y,Prediction);
    gradients = dlgradient(loss, parameters.Weights);

end

function [X, Y] = preprocessMiniBatch(XCell, YCell)
    
    % Extract image data from cell and concatenate
    X = cat(4, XCell{:});
    % Extract label data from cell and concatenate
    Y = cat(4, YCell{:});
        
end

function Y = change_dimension(X)
    
    Y = permute(X, [1 2 4 3]);
    
end
function loss = Myloss1(Y, X)

    loss = mean(reshape((Y - X).^2,1,[]));
    
end

function loss = Myloss(Y, X)

    loss = sum(abs(Y - X), 'all') / size(Y, 4);
    
end

function weights = initializeGlorot(sz, numOut, numIn)

    Z = 2 * rand(sz,'single') - 1;
    bound = sqrt(6 / (numIn + numOut));

    weights = bound * Z;
    weights = dlarray(weights);

end

