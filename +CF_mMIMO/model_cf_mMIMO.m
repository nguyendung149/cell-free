clear all; close all; clc;
%%
format long
% To get the same UE distribution every time you try something new
% rng(4873256)

% Number of APs
L = 16;

% Number of UEs
K = 20;

%Select length of pilot of UEs
tau_p = 10;

%Select length of cohernece block
tau_c = 200;

prelogFactor = (tau_c-tau_p)/(tau_c);

% Number of AP antennas
M = 4;

d0 = 100;
d1 = 500; 
PL = 140.72;

% Select the number of setups with random UE locations
nbrOfSetups = 10;

% Select the number of channel realizations per setup
nbrOfRealizations = 100;

%% Model parameters
% Set the length in meters of the total square area
squareLength = 1000;

% Number of APs per dimension
nbrAPsPerDim = floor(sqrt(L));

% Pathloss exponent
alpha = 3.67;

% Average channel gain in dB at a reference distance of 1 meter.
constantTerm = -30.5;

% Standard deviation of shadow fading
sigma_sf = 1;

% Define the antenna spacing (in number of wavelengths)
antennaSpacing = 1/2; % Half wavelength distance

% Distance between APs in vertical/horizontal direction
interAPDistance = floor(squareLength/nbrAPsPerDim);

%%Propagation parameters

% Communication bandwidth
B = 20e6;

% Total uplink transmit power per UE (mW)
p = 100;

% Maximum downlink transmit power per AP (mW)
Pmax = 1000;

% Compute downlink power per UE in case of equal power allocation
rhoEqual = (Pmax/K)*ones(K,L);

% Square roots of power coefficients for equal power allocation
gammaEqual = sqrt(rhoEqual); 

% Prepare power coefficients for the benchmark in [12]
rho_Giovanni19 = zeros(K,L,nbrOfSetups);

% Vertical distance between APs and UEs
distanceVertical = 10;

% Define noise figure at AP (in dB)
noiseFigure = 7;

% Compute noise power
noiseVariancedBm = -174 + 10*log10(B) + noiseFigure;

% Angular standard deviation in the local scattering model (in degrees)
ASDdeg = 10;

% Store identity matrix of size M x M
eyeM = eye(M);

%% Prepare to save simulation results
% Preallocate SE terms for MR and RZF precoding schemes

% Equal power allocation
SE_MR_equal = zeros(K,nbrOfSetups);
SE_RZF_equal = zeros(K,nbrOfSetups);

% The benchmark in [12]
SE_MR_Giovanni19 = zeros(K,nbrOfSetups);
SE_RZF_Giovanni19 = zeros(K,nbrOfSetups);

% Proposed algorithm with ADMM and WMMSE for sum-SE maximization
SE_MR_WMMSE_ADMM = zeros(K,nbrOfSetups);
SE_RZF_WMMSE_ADMM = zeros(K,nbrOfSetups);

% Proposed algorithm with ADMM and WMMSE for PF maximization
SE_MR_WMMSE_PF_ADMM = zeros(K,nbrOfSetups);
SE_RZF_WMMSE_PF_ADMM = zeros(K,nbrOfSetups);

% DNN allocation
SE_MR_PF_DDNN = zeros(K,nbrOfSetups);
SE_MR_sumSE_DDNN = zeros(K,nbrOfSetups);
SE_RZF_PF_DDNN = zeros(K,nbrOfSetups);
SE_RZF_sumSE_DDNN = zeros(K,nbrOfSetups);

% CDNN allocation
cluster_size = 3;
SE_MR_sumSE_CDNN3 = zeros(K,nbrOfSetups);
SE_RZF_sumSE_CDNN3 = zeros(K,nbrOfSetups);
SE_MR_PF_CDNN3 = zeros(K,nbrOfSetups);
SE_RZF_PF_CDNN3 = zeros(K,nbrOfSetups);

%%
% Prepare array for pilot indices of K UEs for all setups
pilotIndex = zeros(K,1);

% Prepare arrays to save square roots of the power coefficients
% Sum-SE maximization, WMMSE
% ADMM implementation
mu_MR_WMMSE_ADMM = zeros(K,L,nbrOfSetups);
mu_RZF_WMMSE_ADMM = zeros(K,L,nbrOfSetups);

% Prepare arrays for run time
% Sum-SE maximization, WMMSE
% ADMM implementation
stop_MR_WMMSE_ADMM = zeros(nbrOfSetups);
stop_RZF_WMMSE_ADMM = zeros(nbrOfSetups);

%%
% PF maximization, WMMSE
% All arrays initializations
% ADMM implementation
mu_MR_WMMSE_PF_ADMM = zeros(K,L,nbrOfSetups);
mu_RZF_WMMSE_PF_ADMM = zeros(K,L,nbrOfSetups);

% Prepare arrays for run time
% PF maximization, WMMSE
% All arrays initializations
% ADMM implementation
stop_MR_WMMSE_PF_ADMM = zeros(nbrOfSetups);
stop_RZF_WMMSE_PF_ADMM = zeros(nbrOfSetups);

%% Datasets initializations for prediction
dataset_a_MR = zeros(L,K,nbrOfSetups);
dataset_a_RZF = zeros(L,K,nbrOfSetups);

dataset_B_MR = zeros(L,L,K,K,nbrOfSetups);
dataset_B_RZF = zeros(L,L,K,K,nbrOfSetups);

dataset_UEpositions = zeros(K,nbrOfSetups);
dataset_betas = zeros(K,L,nbrOfSetups);
dataset_angletoUE = zeros(K,L,nbrOfSetups);

%%
% Get AP locations and keep them fixed for all the setups
APpositions = load('D:\DuAn\code\new_storage\APpositions.mat');
APXpositions = real(APpositions.APpositions);
APYpositions = imag(APpositions.APpositions);

% Go through each random setup
for n = 1:nbrOfSetups
    % Output simulation progress
    fprintf('%d setups out of %d \n',n,nbrOfSetups)

    UEpositions = zeros(K,1);
    distances = zeros(K,L);

    % Prepare to store normalized spatial correlation matrices
    R = zeros(M,M,K,L);

    % Prepare to store average channel gain numbers (in dB)
    channelGaindB = zeros(K,L);
    
    % Generate random UE locations together
    posXY = squareLength*rand(K,2);

    UEXpositions = posXY(:,1);
    UEYpositions = posXY(:,2);
    UEpositions = UEXpositions + 1i*UEYpositions;
    
    start = tic;

    angletoUE = zeros(K,L);

    for k = 1:K
        Xdist = repmat(UEXpositions(k,1),L,1) - APXpositions;
        Xdistabs = abs(Xdist);
        temp = find(Xdistabs > squareLength/2);
        Xdist(temp,1) = (squareLength - Xdistabs(temp,1)).*sign(-Xdist(temp,1));

        Ydist = repmat(UEYpositions(k,1),L,1) - APYpositions;
        Ydistabs = abs(Ydist);
        temp = find(Ydistabs>squareLength/2);
        Ydist(temp,1) = (squareLength - Ydistabs(temp,1)).*sign(-Ydist(temp,1));

        distances(k,:) = sqrt(distanceVertical.^2 + Xdist(:,1).^2 + Ydist(:,1).^2);
        channelGaindB(k,:) = constantTerm - alpha*10*log10(distances(k,:));
        % Go through all APs
        for j = 1:L
            % if distances(k,j) <= d0
            %     channelGaindB(k,:) = -PL - 35*log10(d1) + 20*log10(d1) - 20*log10(d0);
            % elseif (distances(k,j) >= d0 && distances(k,j) <= d1)
            %     channelGaindB(k,:) = -PL - 35*log10(d1) + 20*log10(d1) - 20*log10(distances(k,:));
            % else
            %     channelGaindB(k,:) = -PL - 35*log10(distances(k,:));
            % end

            % Compute nominal angle between the new UE k and AP l 
            angletoUE(k,j) = angle(Xdist(j) + 1i*Ydist(j));

            R(:,:,k,j) = computedRMatrix(M,angletoUE(k,j),ASDdeg);

        end
       
    end

    t_end = toc - start;
    fprintf('\n\n Time: %f \n',t_end);

    
    channelGainOverNoise = channelGaindB - noiseVariancedBm;
    H = zeros(M,nbrOfRealizations,K,L);
    CH = sqrt(0.5)*(randn(M,nbrOfRealizations,K,L) + 1i*randn(M,nbrOfRealizations,K,L));

    betas = zeros(K,L);
    CorrR = zeros(M,M,K,L);

    for j2 = 1:L
        for k2 = 1:K
            betas(k2,j2) = (10^(channelGainOverNoise(k2,j2)/10));
            CorrR(:,:,k2,j2) = betas(k2,j2) * R(:,:,k2,j2);
            Rsqrt = sqrtm(CorrR(:,:,k2,j2));
            H(:,:,k2,j2) = Rsqrt*CH(:,:,k2,j2);
        end
    end
    %% Perform channel estimation
    % Pilot assignment
    pilotIndex = assignPilots(K,tau_p,betas);

    % Generate realizations of normalized noise
    Np = sqrt(0.5)*(randn(M,nbrOfRealizations,L,tau_p) + 1i*randn(M,nbrOfRealizations,L,tau_p));

    % Prepare to store results
    Hhat = zeros(M,nbrOfRealizations,K,L);
    Hhat_MMSE_MeanSquare = zeros(K,L);

    for l = 1:L
        for t = 1:tau_p
             % Compute processed pilot signal for all UEs that use pilot t
             yp = sqrt(p*tau_p)*sum(H(:,:,t == pilotIndex,l),3) + Np(:,:,l,t);

             % Compute the matrix that is inverted in the MMSE estimator
             PsiInv = (p*tau_p*sum(CorrR(:,:,t==pilotIndex,l),3) + eyeM);

             % Go through all UEs that use pilot t
             arrayPilot = find(t == pilotIndex);
             for k = 1:size(arrayPilot,1)
                RPsi = CorrR(:,:,arrayPilot(k),l)*(PsiInv^-1);
                Hhat(:,:,arrayPilot(k),l) = sqrt(p*tau_p)*(RPsi*yp);
                Hhat_MMSE_MeanSquare(arrayPilot(k),l) = (p*tau_p/M)*real(trace(RPsi*CorrR(:,:,arrayPilot(k),l)));
             end
        end
    end
    w_MR = zeros(M,K,L);
    w_RZF = zeros(M,K,L);

    a_MR = zeros(L,K);
    a_RZF = zeros(L,K);

    B_MR = zeros(L,L,K,K);
    B_RZF = zeros(L,L,K,K);

    interf_MR = zeros(K,K,L);
    interf_RZF = zeros(K,K,L);

    interf2_MR = zeros(K,K,L);
    interf2_RZF = zeros(K,K,L);

    for n1 = 1:nbrOfRealizations
        for j3 = 1:L
            V_MR = reshape(Hhat(:,n1,:,j3),4,20);
            V_RZF = (p*V_MR*conj(V_MR).' +eyeM)^(-1)*V_MR;
            normV_MR = [];
            normV_RZF = [];
            for y = 1:size(V_MR,2)
                a = norm(V_MR(:,y,:),"fro");
                normV_MR = [normV_MR a];
            end
            w_MR(:,:,j3) = V_MR./normV_MR;

            for y = 1:size(V_RZF,2)
                a = norm(V_RZF(:,y,:),"fro");
                normV_RZF = [normV_RZF a];
            end
            w_RZF(:,:,j3) = V_RZF./normV_RZF;
        end

        for j4 = 1:L
            for k4 = 1:K
                a_MR(j4,k4) = a_MR(j4,k4) +  (conj(H(:,n1,k4,j4)).'*(w_MR(:,k4,j4))/nbrOfRealizations);
                a_RZF(j4,k4) = a_RZF(j4,k4) + (conj(H(:,n1,k4,j4)).'*(w_RZF(:,k4,j4))/nbrOfRealizations);
                for i4 = 1:K
                    interf_MR(k4,i4,j4) = interf_MR(k4,i4,j4) + (conj(H(:,n1,k4,j4)).'*w_MR(:,i4,j4))/nbrOfRealizations;
                    interf_RZF(k4,i4,j4) = interf_RZF(k4,i4,j4) + (conj(H(:,n1,k4,j4)).'*w_RZF(:,i4,j4))/nbrOfRealizations;
                    
                    interf2_MR(k4,i4,j4) = interf2_MR(k4,i4,j4) + (abs((conj(H(:,n1,k4,j4)).'*w_MR(:,i4,j4)))^2)/nbrOfRealizations;
                    interf2_RZF(k4,i4,j4) = interf2_RZF(k4,i4,j4) + (abs((conj(H(:,n1,k4,j4)).'*w_RZF(:,i4,j4)))^2)/nbrOfRealizations;


                end
            end
            
        end
    end
    for k5 = 1:K
        for i5 = 1:K
            B_MR(:,:,k5,i5) = reshape(interf_MR(k5,i5,:),L,1)*reshape(conj(interf_MR(k5,i5,:)),1,L);
            B_RZF(:,:,k5,i5) = reshape(interf_RZF(k5,i5,:),L,1)*reshape(conj(interf_RZF(k5,i5,:)),1,L);

        end 
    end
    for j5 = 1:L
        rho_Giovanni19(:,j5,n) = Pmax*(sqrt(Hhat_MMSE_MeanSquare(:,j5)))/(sum(sqrt(Hhat_MMSE_MeanSquare(:,j5))));
        B_MR(j5,j5,:,:) = interf2_MR(:,:,j5);
        B_RZF(j5,j5,:,:) = interf2_RZF(:,:,j5); 
    end
    
    a_MR = abs(a_MR);
    a_RZF = abs(a_RZF);

    B_MR = real(B_MR);
    B_RZF = real(B_RZF);

    dataset_a_MR(:,:,n) = a_MR;
    dataset_a_RZF(:,:,n) = a_RZF;
    dataset_B_MR(:,:,:,:,n) = B_MR;
    dataset_B_RZF(:,:,:,:,n) = B_RZF;
    dataset_UEpositions(:,n) = UEpositions(:,1);
    dataset_betas(:,:,n) = betas;
    dataset_angletoUE(:,:,n) = angletoUE;

    SE_MR_equal(:,n) = calculate_SINR_and_SE_DL(a_MR,B_MR,prelogFactor,gammaEqual,Pmax);
    SE_RZF_equal(:,n) = calculate_SINR_and_SE_DL(a_RZF,B_RZF,prelogFactor,gammaEqual,Pmax);
    SE_MR_Giovanni19(:,n) = calculate_SINR_and_SE_DL(a_MR,B_MR,prelogFactor,sqrt(rho_Giovanni19(:,:,n)),Pmax);
    SE_RZF_Giovanni19(:,n) = calculate_SINR_and_SE_DL(a_RZF,B_RZF,prelogFactor,sqrt(rho_Giovanni19(:,:,n)),Pmax);
    
    % Sum-SE ADMM
    fprintf('WMMSE ADMM\n')
    start_MR_WMMSE = tic;
    mu_MR_WMMSE_ADMM(:,:,n) = WMMSE_ADMM_timing(L,K,Pmax,a_MR,B_MR);
    stop_MR_WMMSE_ADMM(n) = toc - start_MR_WMMSE;
    SE_MR_WMMSE_ADMM(:,n) = calculate_SINR_and_SE_DL(a_MR,B_MR,prelogFactor,mu_MR_WMMSE_ADMM(:,:,n),Pmax);

    start_RZF_WMMSE = tic;
    mu_RZF_WMMSE_ADMM(:,:,n) = WMMSE_ADMM_timing(L, K, Pmax, a_RZF, B_RZF);
    stop_RZF_WMMSE_ADMM(n) = toc - start_RZF_WMMSE;
    SE_RZF_WMMSE_ADMM(:,n) = calculate_SINR_and_SE_DL(a_RZF,B_RZF,prelogFactor,mu_RZF_WMMSE_ADMM(:,:,n),Pmax);

    % Proportional Fairness ADMM
    fprintf('WMMSE PF ADMM\n')
    start_MR_WMMSE = tic;
    mu_MR_WMMSE_PF_ADMM(:,:,n) = WMMSE_ADMM_timing_PF(L, K, Pmax, a_MR, B_MR);
    stop_MR_WMMSE_PF_ADMM(n) = toc - start_MR_WMMSE;
    SE_MR_WMMSE_PF_ADMM(:,n) = calculate_SINR_and_SE_DL(a_MR,B_MR,prelogFactor,mu_MR_WMMSE_PF_ADMM(:,:,n), Pmax);
    
    start_RZF_WMMSE = tic;
    mu_RZF_WMMSE_PF_ADMM(:,:,n) = WMMSE_ADMM_timing_PF(L, K, Pmax, a_RZF, B_RZF);
    stop_RZF_WMMSE_PF_ADMM(n) = toc - start_RZF_WMMSE;
    SE_RZF_WMMSE_PF_ADMM(:,n) = calculate_SINR_and_SE_DL(a_RZF,B_RZF,prelogFactor,mu_RZF_WMMSE_PF_ADMM(:,:,n), Pmax);



end

dataset_B_MR = real(dataset_B_MR);
dataset_B_RZF = real(dataset_B_RZF);

% sumSE DDNN
model = load(".\Model\Su dung Robust Scaler\MR_sumSE_DDNN");
muMR_DDNN_sumSE = pred_func(dataset_betas,Pmax,nbrOfSetups,model.modelArray);
muMR_DDNN_sumSE_scaling = muMR_DDNN_sumSE(K+1,:,:);
muMR_DDNN_sumSE_scaling(muMR_DDNN_sumSE_scaling > 1) = 1;
muMR_DDNN_sumSE_scaling = muMR_DDNN_sumSE_scaling * sqrt(Pmax);

for n = 1:nbrOfSetups
    normMuMR_DDNN_sumSEArray = [];
    tem = reshape(muMR_DDNN_sumSE(1:K,:,n),K,[]);
    for t = 1:size(tem,2)
        value = norm(tem(:,t),"fro");
        normMuMR_DDNN_sumSEArray = [normMuMR_DDNN_sumSEArray value];    
    end
    muMR_DDNN_sumSE(1:K,:,n) = muMR_DDNN_sumSE(1:K,:,n) .* repmat(muMR_DDNN_sumSE_scaling(:,:,n),K,1) ./ max(normMuMR_DDNN_sumSEArray);
end

for n = 1:nbrOfSetups
    SE_MR_sumSE_DDNN(:,n) = calculate_SINR_and_SE_DL(dataset_a_MR(:,:,n), dataset_B_MR(:,:,:,:,n), prelogFactor, muMR_DDNN_sumSE(1:K,:,n), Pmax);
end

model = load(".\Model\Su dung Robust Scaler\RZF_sumSE_DDNN.mat");
muRZF_DDNN_sumSE = pred_func(dataset_betas,Pmax,nbrOfSetups,model.modelArray);
muRZF_DDNN_sumSE_scaling = muRZF_DDNN_sumSE(K+1,:,:);
muRZF_DDNN_sumSE_scaling(muRZF_DDNN_sumSE_scaling > 1) = 1;
muRZF_DDNN_sumSE_scaling = muRZF_DDNN_sumSE_scaling * sqrt(Pmax);

for n = 1:nbrOfSetups
    normMuRZF_DDNN_sumSEArray = [];
    tem = reshape(muRZF_DDNN_sumSE(1:K,:,n),K,[]);
    for t = 1:size(tem,2)
        value = norm(tem(:,t),"fro");
        normMuRZF_DDNN_sumSEArray = [normMuRZF_DDNN_sumSEArray value];    
    end
    muRZF_DDNN_sumSE(1:K,:,n) = muRZF_DDNN_sumSE(1:K,:,n) .* repmat(muRZF_DDNN_sumSE_scaling(:,:,n),K,1) ./ max(normMuRZF_DDNN_sumSEArray);
end

for n = 1:nbrOfSetups
    SE_RZF_sumSE_DDNN(:,n) = calculate_SINR_and_SE_DL(dataset_a_RZF(:,:,n), dataset_B_RZF(:,:,:,:,n), prelogFactor, muRZF_DDNN_sumSE(1:K,:,n), Pmax);
end

% PF DDNN
model = load(".\Model\Su dung Robust Scaler\MR_PF_DDNN.mat");
muMR_DDNN_PF = pred_func(dataset_betas,Pmax,nbrOfSetups,model.modelArray);
muMR_DDNN_PF_scaling = muMR_DDNN_PF(K+1,:,:);
muMR_DDNN_PF_scaling(muMR_DDNN_PF_scaling > 1) = 1;
muMR_DDNN_PF_scaling = muMR_DDNN_PF_scaling * sqrt(Pmax);

for n = 1:nbrOfSetups
    normMuMR_DDNN_PFArray = [];
    tem = reshape(muMR_DDNN_PF(1:K,:,n),K,[]);
    for t = 1:size(tem,2)
        value = norm(tem(:,t),"fro");
        normMuMR_DDNN_PFArray = [normMuMR_DDNN_PFArray value];    
    end
    muMR_DDNN_PF(1:K,:,n) = muMR_DDNN_PF(1:K,:,n) .* repmat(muMR_DDNN_PF_scaling(:,:,n),K,1) ./ max(normMuMR_DDNN_PFArray);
end

for n = 1:nbrOfSetups
    SE_MR_PF_DDNN(:,n) = calculate_SINR_and_SE_DL(dataset_a_MR(:,:,n), dataset_B_MR(:,:,:,:,n), prelogFactor, muMR_DDNN_PF(1:K,:,n), Pmax);
end
model = load(".\Model\Su dung Robust Scaler\RZF_PF_DDNN.mat");
muRZF_DDNN_PF = pred_func(dataset_betas,Pmax,nbrOfSetups,model.modelArray);
muRZF_DDNN_PF_scaling = muRZF_DDNN_PF(K+1,:,:);
muRZF_DDNN_PF_scaling(muRZF_DDNN_PF_scaling > 1) = 1;
muRZF_DDNN_PF_scaling = muRZF_DDNN_PF_scaling * sqrt(Pmax);

for n = 1:nbrOfSetups
    normMuRZF_DDNN_PFArray = [];
    tem = reshape(muRZF_DDNN_PF(1:K,:,n),K,[]);
    for t = 1:size(tem,2)
        value = norm(tem(:,t),"fro");
        normMuRZF_DDNN_PFArray = [normMuRZF_DDNN_PFArray value];    
    end
    muRZF_DDNN_PF(1:K,:,n) = muRZF_DDNN_PF(1:K,:,n) .* repmat(muRZF_DDNN_PF_scaling(:,:,n),K,1) ./ max(normMuRZF_DDNN_PFArray);
end

for n = 1:nbrOfSetups
    SE_RZF_PF_DDNN(:,n) = calculate_SINR_and_SE_DL(dataset_a_RZF(:,:,n), dataset_B_RZF(:,:,:,:,n), prelogFactor, muRZF_DDNN_PF(1:K,:,n), Pmax);
end
% sumSE CDNN
model = load(".\Model\CDNN\MR_SE_CDNN.mat");
muMR_CDNN_sumSE = predictions_CDNN(dataset_betas,Pmax,nbrOfSetups,model.modelArray,cluster_size);
muMR_CDNN_sumSE_scaling = muMR_CDNN_sumSE(K+1,:,:);
muMR_CDNN_sumSE_scaling(muMR_CDNN_sumSE_scaling > 1) = 1;
muMR_CDNN_sumSE_scaling = muMR_CDNN_sumSE_scaling * sqrt(Pmax);
for n = 1:nbrOfSetups
    normMuMR_CDNN_sumSEArray = [];
    tem = reshape(muMR_CDNN_sumSE(1:K,:,n),K,[]);
    for t = 1:size(tem,2)
        value = norm(tem(:,t),"fro");
        normMuMR_CDNN_sumSEArray = [normMuMR_CDNN_sumSEArray value];    
    end
    muMR_CDNN_sumSE(1:K,:,n) = muMR_CDNN_sumSE(1:K,:,n) .* repmat(muMR_CDNN_sumSE_scaling(:,:,n),K,1) ./ max(normMuMR_CDNN_sumSEArray);
end

for n = 1:nbrOfSetups
    SE_MR_sumSE_CDNN3(:,n) = calculate_SINR_and_SE_DL(dataset_a_MR(:,:,n), dataset_B_MR(:,:,:,:,n), prelogFactor, muMR_CDNN_sumSE(1:K,:,n), Pmax);
end
model = load(".\Model\CDNN\RZF_SE_CDNN.mat");
muRZF_CDNN_sumSE = predictions_CDNN(dataset_betas,Pmax,nbrOfSetups,model.modelArray,cluster_size);
muRZF_CDNN_sumSE_scaling = muRZF_CDNN_sumSE(K+1,:,:);
muRZF_CDNN_sumSE_scaling(muRZF_CDNN_sumSE_scaling > 1) = 1;
muRZF_CDNN_sumSE_scaling = muRZF_CDNN_sumSE_scaling * sqrt(Pmax);
for n = 1:nbrOfSetups
    normMuRZF_CDNN_sumSEArray = [];
    tem = reshape(muRZF_CDNN_sumSE(1:K,:,n),K,[]);
    for t = 1:size(tem,2)
        value = norm(tem(:,t),"fro");
        normMuRZF_CDNN_sumSEArray = [normMuRZF_CDNN_sumSEArray value];    
    end
    muRZF_CDNN_sumSE(1:K,:,n) = muRZF_CDNN_sumSE(1:K,:,n) .* repmat(muRZF_CDNN_sumSE_scaling(:,:,n),K,1) ./ max(normMuRZF_CDNN_sumSEArray);
end

for n = 1:nbrOfSetups
    SE_RZF_sumSE_CDNN3(:,n) = calculate_SINR_and_SE_DL(dataset_a_RZF(:,:,n), dataset_B_RZF(:,:,:,:,n), prelogFactor, muRZF_CDNN_sumSE(1:K,:,n), Pmax);
end

% PF CDNN
model = load(".\Model\CDNN\MR_PF_CDNN.mat");
muMR_CDNN_PF = predictions_CDNN(dataset_betas,Pmax,nbrOfSetups,model.modelArray,cluster_size);
muMR_CDNN_PF_scaling = muMR_CDNN_PF(K+1,:,:);
muMR_CDNN_PF_scaling(muMR_CDNN_PF_scaling > 1) = 1;
muMR_CDNN_PF_scaling = muMR_CDNN_PF_scaling * sqrt(Pmax);
for n = 1:nbrOfSetups
    normMuMR_CDNN_PFArray = [];
    tem = reshape(muMR_CDNN_PF(1:K,:,n),K,[]);
    for t = 1:size(tem,2)
        value = norm(tem(:,t),"fro");
        normMuMR_CDNN_PFArray = [normMuMR_CDNN_PFArray value];    
    end
    muMR_CDNN_PF(1:K,:,n) = muMR_CDNN_PF(1:K,:,n) .* repmat(muMR_CDNN_PF_scaling(:,:,n),K,1) ./ max(normMuMR_CDNN_PFArray);
end

for n = 1:nbrOfSetups
    SE_MR_PF_CDNN3(:,n) = calculate_SINR_and_SE_DL(dataset_a_MR(:,:,n), dataset_B_MR(:,:,:,:,n), prelogFactor, muMR_CDNN_PF(1:K,:,n), Pmax);
end

model = load(".\Model\CDNN\RZF_PF_CDNN.mat");
muRZF_CDNN_PF = predictions_CDNN(dataset_betas,Pmax,nbrOfSetups,model.modelArray,cluster_size);
muRZF_CDNN_PF_scaling = muRZF_CDNN_PF(K+1,:,:);
muRZF_CDNN_PF_scaling(muRZF_CDNN_PF_scaling > 1) = 1;
muRZF_CDNN_PF_scaling = muRZF_CDNN_PF_scaling * sqrt(Pmax);
for n = 1:nbrOfSetups
    normMuRZF_CDNN_PFArray = [];
    tem = reshape(muRZF_CDNN_PF(1:K,:,n),K,[]);
    for t = 1:size(tem,2)
        value = norm(tem(:,t),"fro");
        normMuRZF_CDNN_PFArray = [normMuRZF_CDNN_PFArray value];    
    end
    muRZF_CDNN_PF(1:K,:,n) = muRZF_CDNN_PF(1:K,:,n) .* repmat(muRZF_CDNN_PF_scaling(:,:,n),K,1) ./ max(normMuRZF_CDNN_PFArray);
end

for n = 1:nbrOfSetups
    SE_RZF_PF_CDNN3(:,n) = calculate_SINR_and_SE_DL(dataset_a_RZF(:,:,n), dataset_B_RZF(:,:,:,:,n), prelogFactor, muRZF_CDNN_PF(1:K,:,n), Pmax);
end

% Sort the SE values for CDF plots
sorted_SE_MR_Equal = sorted_SE(SE_MR_equal);
sorted_SE_RZF_Equal = sorted_SE(SE_RZF_equal);
sorted_SE_MR_Giovanni19 = sorted_SE(SE_MR_Giovanni19);
sorted_SE_RZF_Giovanni19 = sorted_SE(SE_RZF_Giovanni19);
sorted_SE_MR_WMMSE_ADMM = sorted_SE(SE_MR_WMMSE_ADMM);
sorted_SE_RZF_WMMSE_ADMM = sorted_SE(SE_RZF_WMMSE_ADMM);
sorted_SE_MR_WMMSE_PF_ADMM = sorted_SE(SE_MR_WMMSE_PF_ADMM);
sorted_SE_RZF_WMMSE_PF_ADMM = sorted_SE(SE_RZF_WMMSE_PF_ADMM);


sorted_SE_MR_sumSE_DDNN = sorted_SE(SE_MR_sumSE_DDNN);
sorted_SE_RZF_sumSE_DDNN = sorted_SE(SE_RZF_sumSE_DDNN);
sorted_SE_MR_PF_DDNN = sorted_SE(SE_MR_PF_DDNN);
sorted_SE_RZF_PF_DDNN = sorted_SE(SE_RZF_PF_DDNN);

sorted_SE_MR_sumSE_CDNN3 = sorted_SE(SE_MR_sumSE_CDNN3);
sorted_SE_RZF_sumSE_CDNN3 = sorted_SE(SE_RZF_sumSE_CDNN3);
sorted_SE_MR_PF_CDNN3 = sorted_SE(SE_MR_PF_CDNN3);
sorted_SE_RZF_PF_CDNN3 = sorted_SE(SE_RZF_PF_CDNN3);


% Calculations for sum-rate plots
sum_SE_MR_Equal = sum(SE_MR_equal);
sum_SE_MR_Equal = sort(sum_SE_MR_Equal);
sum_SE_RZF_Equal = sum(SE_RZF_equal);
sum_SE_RZF_Equal = sort(sum_SE_RZF_Equal);

sum_SE_MR_Giovanni19 = sum(SE_MR_Giovanni19);
sum_SE_MR_Giovanni19 = sort(sum_SE_MR_Giovanni19);
sum_SE_RZF_Giovanni19 = sum(SE_RZF_Giovanni19);
sum_SE_RZF_Giovanni19 = sort(sum_SE_RZF_Giovanni19);

sum_SE_MR_WMMSE_ADMM = sum(SE_MR_WMMSE_ADMM);
sum_SE_MR_WMMSE_ADMM = sort(sum_SE_MR_WMMSE_ADMM);
sum_SE_RZF_WMMSE_ADMM = sum(SE_RZF_WMMSE_ADMM);
sum_SE_RZF_WMMSE_ADMM = sort(sum_SE_RZF_WMMSE_ADMM);

sum_SE_MR_WMMSE_PF_ADMM = sum(SE_MR_WMMSE_PF_ADMM);
sum_SE_MR_WMMSE_PF_ADMM = sort(sum_SE_MR_WMMSE_PF_ADMM);
sum_SE_RZF_WMMSE_PF_ADMM = sum(SE_RZF_WMMSE_PF_ADMM);
sum_SE_RZF_WMMSE_PF_ADMM = sort(sum_SE_RZF_WMMSE_PF_ADMM);

sum_SE_MR_sumSE_DDNN = sum(SE_MR_sumSE_DDNN);
sum_SE_MR_sumSE_DDNN = sort(sum_SE_MR_sumSE_DDNN);
sum_SE_RZF_sumSE_DDNN = sum(SE_RZF_sumSE_DDNN);
sum_SE_RZF_sumSE_DDNN = sort(sum_SE_RZF_sumSE_DDNN);

sum_SE_MR_PF_DDNN = sum(SE_MR_PF_DDNN);
sum_SE_MR_PF_DDNN = sort(sum_SE_MR_PF_DDNN);
sum_SE_RZF_PF_DDNN = sum(SE_RZF_PF_DDNN);
sum_SE_RZF_PF_DDNN = sort(sum_SE_RZF_PF_DDNN);

sum_SE_MR_sumSE_CDNN3 = sum(SE_MR_sumSE_CDNN3);
sum_SE_MR_sumSE_CDNN3 = sort(sum_SE_MR_sumSE_CDNN3);
sum_SE_RZF_sumSE_CDNN3 = sum(SE_RZF_sumSE_CDNN3);
sum_SE_RZF_sumSE_CDNN3 = sort(sum_SE_RZF_sumSE_CDNN3);

sum_SE_MR_PF_CDNN3 = sum(SE_MR_PF_CDNN3);
sum_SE_MR_PF_CDNN3 = sort(sum_SE_MR_PF_CDNN3);
sum_SE_RZF_PF_CDNN3 = sum(SE_RZF_PF_CDNN3);
sum_SE_RZF_PF_CDNN3 = sort(sum_SE_RZF_PF_CDNN3);


% sum_SE_MR_sumSE_ANN = sum(SE_MR_sumSE_ANN);
% sum_SE_MR_sumSE_ANN = sort(sum_SE_MR_sumSE_ANN);
% sum_SE_RZF_sumSE_ANN = sum(SE_RZF_sumSE_ANN);
% sum_SE_RZF_sumSE_ANN = sort(sum_SE_RZF_sumSE_ANN);
% 
% 
% sum_SE_MR_PF_ANN = sum(SE_MR_PF_ANN);
% sum_SE_MR_PF_ANN = sort(sum_SE_MR_PF_ANN);
% sum_SE_RZF_PF_ANN = sum(SE_RZF_PF_ANN);
% sum_SE_RZF_PF_ANN = sort(sum_SE_RZF_PF_ANN);


%% Plot
Yvals = linspace(0,1,K*nbrOfSetups);
Yvals_sum = linspace(0,1,nbrOfSetups);

figure(1)
hold on
grid on
plot(sorted_SE_MR_Equal,  Yvals, color="red",  linewidth=2)
plot(sorted_SE_MR_Giovanni19, Yvals, '-.', color="green",  linewidth=2)
plot(sorted_SE_MR_WMMSE_ADMM, Yvals, color="blue", linewidth=2)
plot(sorted_SE_MR_sumSE_DDNN, Yvals, color="cyan", linewidth=2)
% plot(sorted_SE_MR_sumSE_ANN, Yvals, color="magenta",  linewidth=2)
plot(sorted_SE_MR_sumSE_CDNN3,Yvals,color="yellow",LineWidth=2)


legend('Equal power MR','[12] MR','SE MR WMMSE ADMM', 'SE MR sumSE DDNN',...
    'SE MR sumSE CDNN3')
xlim([0.0, 5])
ylim([0, 1])
title('Sorted Spectral Efficiency')
xlabel('SE per UE [bit/s/Hz]')
ylabel('CDF')

figure(2)
hold on
grid on
plot(sorted_SE_MR_Equal,  Yvals, color="red",  linewidth=2)
plot(sorted_SE_MR_Giovanni19, Yvals, '-.', color="green",  linewidth=2)
plot(sorted_SE_MR_WMMSE_PF_ADMM, Yvals, color="blue", linewidth=2)
plot(sorted_SE_MR_PF_DDNN, Yvals, color="cyan", linewidth=2) 
%plot(sorted_SE_MR_PF_ANN, Yvals, color="magenta",linewidth=2)
plot(sorted_SE_MR_PF_CDNN3,Yvals,color="yellow",LineWidth=2)

legend('Equal power MR','[12] MR','SE MR WMMSE PF ADMM','SE MR PF DDNN',...
    'SE MR PF CDNN3')
xlim([0.0, 5])
ylim([0, 1])
title('Sorted Spectral Efficiency')
xlabel('SE per UE [bit/s/Hz]')
ylabel('CDF')

figure(3)
hold on
grid on
plot(sorted_SE_RZF_Equal,  Yvals, color="red",  linewidth=2)
plot(sorted_SE_RZF_Giovanni19, Yvals, '-.', color="green",  linewidth=2)
plot(sorted_SE_RZF_WMMSE_ADMM, Yvals, color="blue", linewidth=2)
plot(sorted_SE_RZF_sumSE_DDNN, Yvals, color="cyan", linewidth=2)
% plot(sorted_SE_RZF_sumSE_ANN, Yvals, color="magenta", linewidth=2)
plot(sorted_SE_RZF_sumSE_CDNN3,Yvals,color="yellow",LineWidth=2)

legend('Equal power RZF','[12] RZF','SE RZF WMMSE ADMM','SE RZF sumSE DDNN',...
    'SE RZF sumSE CDNN3');
xlim([0.0, 5])
ylim([0, 1])
title('Sorted Spectral Efficiency')
xlabel('SE per UE [bit/s/Hz]')
ylabel('CDF')

figure(4)
hold on
grid on
plot(sorted_SE_RZF_Equal,  Yvals, color="red",  linewidth=2)
plot(sorted_SE_RZF_Giovanni19, Yvals, '-.', color="green",  linewidth=2)
plot(sorted_SE_RZF_WMMSE_PF_ADMM, Yvals, color="blue", linewidth=2) 
plot(sorted_SE_RZF_PF_DDNN, Yvals, color="cyan", linewidth=2) 
% plot(sorted_SE_RZF_PF_ANN, Yvals, color="magenta", linewidth=2)
plot(sorted_SE_RZF_PF_CDNN3, Yvals, color="yellow", linewidth=2) 

legend('Equal power RZF','[12] RZF','SE RZF WMMSE PF ADMM','SE RZF PF DDNN',...
    'SE RZF PF CDNN3')
xlim([0.0, 5])
ylim([0, 1])
title('Sorted Spectral Efficiency')
xlabel('SE per UE [bit/s/Hz]')
ylabel('CDF')

figure(5)
hold on
grid on
plot(sum_SE_MR_Equal,  Yvals_sum, color="red",  linewidth=2)
plot(sum_SE_MR_Giovanni19, Yvals_sum, '-.', color="green",  linewidth=2)
plot(sum_SE_MR_WMMSE_ADMM, Yvals_sum, color="blue", linewidth=2)
plot(sum_SE_MR_sumSE_DDNN, Yvals_sum, color="cyan", linewidth=2) 
% plot(sum_SE_MR_sumSE_ANN, Yvals_sum, color="magenta", linewidth=2)
plot(sum_SE_MR_sumSE_CDNN3,Yvals_sum,color="yellow",LineWidth=2)

legend('Equal power MR','[12] MR','SE MR WMMSE ADMM',...
    'SE MR sumSE DDNN','SE MR sumSE CDNN3')
%xlim([0.0, 5])
ylim([0, 1])
title('Sum Spectral Efficiency')
xlabel('Total SE [bit/s/Hz]')
ylabel('CDF')

figure(6)
hold on
grid on
plot(sum_SE_MR_Equal,  Yvals_sum, color="red",  linewidth=2)
plot(sum_SE_MR_Giovanni19, Yvals_sum, '-.', color="green",  linewidth=2)
plot(sum_SE_MR_WMMSE_PF_ADMM, Yvals_sum, color="blue", linewidth=2)
plot(sum_SE_MR_PF_DDNN, Yvals_sum, color="cyan", linewidth=2) 
% plot(sum_SE_MR_PF_ANN,Yvals_sum,color="magenta",linewidth=2)
plot(sum_SE_MR_PF_CDNN3, Yvals_sum, color="yellow", linewidth=2) 
legend('Equal power MR','[12] MR','SE MR WMMSE PF ADMM','SE MR PF DDNN',...
    'SE MR PF CDNN3')
%xlim([0.0, 5])
ylim([0, 1])
title('Sum Spectral Efficiency')
xlabel('Total SE [bit/s/Hz]')
ylabel('CDF')

figure(7)
hold on
grid on
plot(sum_SE_RZF_Equal,  Yvals_sum, color="red",  linewidth=2)
plot(sum_SE_RZF_Giovanni19, Yvals_sum, '-.', color="green",  linewidth=2)
plot(sum_SE_RZF_WMMSE_ADMM, Yvals_sum, color="blue", linewidth=2) 
plot(sum_SE_RZF_sumSE_DDNN, Yvals_sum, color="cyan", linewidth=2) 
%plot(sum_SE_RZF_sumSE_ANN, Yvals_sum, color="magenta", linewidth=2)
plot(sum_SE_RZF_sumSE_CDNN3,Yvals_sum,color="yellow",LineWidth=2)

legend('Equal power RZF','[12] RZF','SE RZF WMMSE ADMM','SE RZF sumSE DDNN',...
    'SE RZF sumSE CDNN3')
%xlim([0.0, 5])
ylim([0, 1])
title('Sum Spectral Efficiency')
xlabel('Total SE [bit/s/Hz]')
ylabel('CDF')

figure(8)
hold on
grid on
plot(sum_SE_RZF_Equal,  Yvals_sum, color="red",  linewidth=2)
plot(sum_SE_RZF_Giovanni19, Yvals_sum, '-.', color="green",  linewidth=2)
plot(sum_SE_RZF_WMMSE_PF_ADMM, Yvals_sum, color="blue", linewidth=2)
plot(sum_SE_RZF_PF_DDNN, Yvals_sum, color="cyan", linewidth=2) 
%plot(sum_SE_RZF_PF_ANN, Yvals_sum, color="magenta",linewidth=2)
plot(sum_SE_RZF_PF_CDNN3, Yvals_sum, color="yellow", linewidth=2) 

legend('Equal power RZF','[12] RZF','SE RZF WMMSE PF ADMM',...
    'SE RZF PF DDNN','SE RZF PF CDNN3')
%xlim([0.0, 5])
ylim([0, 1])
title('Sum Spectral Efficiency')
xlabel('Total SE [bit/s/Hz]')
ylabel('CDF')