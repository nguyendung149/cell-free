SNR_Range = 0:5:25;
Num_of_frame_each_SNR = 5000;

% Import Hybrid Network
load('parameters_100.mat');

% Import Hybrid Network
load('parameters_100_10.mat');

% Import Hybrid Network
load('parameters_100_20.mat');

% Import Hybrid Network
load('parameters_100_30.mat');

% Import Hybrid Network
load('parameters_100_50.mat');

% Import Hybrid Network
load('parameters_100_70.mat');

MSE_LS_over_SNR = zeros(length(SNR_Range), 1);
MSE_Hybrid_over_SNR = zeros(length(SNR_Range), 1);
MSE_Hybrid_10_over_SNR = zeros(length(SNR_Range), 1);
MSE_Hybrid_20_over_SNR = zeros(length(SNR_Range), 1);
MSE_Hybrid_30_over_SNR = zeros(length(SNR_Range), 1);
MSE_Hybrid_50_over_SNR = zeros(length(SNR_Range), 1);
MSE_Hybrid_70_over_SNR = zeros(length(SNR_Range), 1);

for SNR = SNR_Range
    
M = 4; % QPSK
k = log2(M);

Parameter.parameters

Num_of_QPSK_symbols = Num_of_FFT * Num_of_symbols * Num_of_frame_each_SNR;
Num_of_bits = Num_of_QPSK_symbols * k;

LS_MSE_in_frame = zeros(Num_of_frame_each_SNR, 1);
MMSE_MSE_in_frame = zeros(Num_of_frame_each_SNR, 1);
DNN_MSE_in_frame = zeros(Num_of_frame_each_SNR, 1);
Transformer_MSE_in_frame = zeros(Num_of_frame_each_SNR, 1);
Hybrid_MSE_in_frame = zeros(Num_of_frame_each_SNR, 1);
Hybrid_MSE_in_frame_10 = zeros(Num_of_frame_each_SNR, 1);
Hybrid_MSE_in_frame_20 = zeros(Num_of_frame_each_SNR, 1);
Hybrid_MSE_in_frame_30 = zeros(Num_of_frame_each_SNR, 1);
Hybrid_MSE_in_frame_50 = zeros(Num_of_frame_each_SNR, 1);
Hybrid_MSE_in_frame_70 = zeros(Num_of_frame_each_SNR, 1);

for Frame = 1 : Num_of_frame_each_SNR

% Data generation
N = Num_of_FFT * Num_of_symbols;
data = randi([0 1], N, k);
Data = reshape(data, [], 1);
dataSym = bi2de(data);

% QPSK modulator
QPSK_symbol = OFDM.QPSK_Modualtor(dataSym);
QPSK_signal = reshape(QPSK_symbol, Num_of_FFT, Num_of_symbols);

% Pilot inserted
[data_in_IFFT, data_location] = OFDM.Pilot_Insert(Pilot_value_user, Pilot_location_symbols, Pilot_location, Frame_size, Num_of_FFT, QPSK_signal);
[data_for_channel, ~] = OFDM.Pilot_Insert(1, Pilot_location_symbols, kron((1 : Num_of_FFT)', ones(1, Num_of_pilot)), Frame_size, Num_of_FFT, (ones(Num_of_FFT, Num_of_symbols)));

% OFDM Transmitter
[Transmitted_signal, ~] = OFDM.OFDM_Transmitter(data_in_IFFT, Num_of_FFT, length_of_CP);
[Transmitted_signal_for_channel, ~] = OFDM.OFDM_Transmitter(data_for_channel, Num_of_FFT, length_of_CP);

% Channel
 
SNR_OFDM = SNR + 10 * log10((Num_of_subcarriers / Num_of_FFT));
Doppler_shift = randi([0, MaxDopplerShift]);
[Multitap_Channel_Signal, Multitap_Channel_Signal_user, Multitap_Channel_Signal_user_for_channel] = Channel.Propagation_Channel_Model(Transmitted_signal, Transmitted_signal_for_channel, ...
    SNR_OFDM, SampleRate, Carrier_Frequency, PathDelays, AveragePathGains, Doppler_shift, DelayProfile);

% OFDM Receiver
[Unrecovered_signal, RS_User] = OFDM.OFDM_Receiver(Multitap_Channel_Signal, Num_of_FFT, length_of_CP, length_of_symbol, Multitap_Channel_Signal_user);
[~, RS] = OFDM.OFDM_Receiver(Multitap_Channel_Signal_user_for_channel, Num_of_FFT, length_of_CP, length_of_symbol, Multitap_Channel_Signal_user_for_channel);

[Received_pilot, ~] = OFDM.Pilot_extract(RS_User, Pilot_location, Num_of_pilot, Pilot_location_symbols, data_location);
H_Ref = Received_pilot ./ Pilot_value_user;

% Channel estimation

[Received_pilot_LS, ~] = OFDM.Pilot_extract(Unrecovered_signal, Pilot_location, Num_of_pilot, Pilot_location_symbols, data_location);

H_LS = CSI.LS(Received_pilot_LS, Pilot_value_user);

H_LS_frame = imresize(H_LS, [Num_of_FFT, max(Pilot_location_symbols)]);
H_LS_frame(:, max(Pilot_location_symbols) + 1 : Frame_size) = kron(H_LS_frame(:, max(Pilot_location_symbols)), ones(1, Frame_size - max(Pilot_location_symbols)));

MSE_LS_frame = mean(abs(H_LS_frame - RS).^2, 'all');

% Hybrid structure

Feature_signal(:, 1, 1) = reshape(real(H_LS), [], 1);
Feature_signal(:, 2, 1) = reshape(imag(H_LS), [], 1);
Feature_signal = dlarray(Feature_signal);

H_Hybrid_feature = transformer.model(Feature_signal, parameters);

H_Hybrid_frame = reshape(extractdata(H_Hybrid_feature(:, 1)), Num_of_FFT, Frame_size) + 1j * reshape(extractdata(H_Hybrid_feature(:, 2)), Num_of_FFT, Frame_size);

MSE_Hybrid_frame = mean(abs(H_Hybrid_frame - RS).^2, 'all');

% Hybrid structure 10% pruning

H_Hybrid_feature = transformer.model(Feature_signal, parameters_10);

H_Hybrid_frame = reshape(extractdata(H_Hybrid_feature(:, 1)), Num_of_FFT, Frame_size) + 1j * reshape(extractdata(H_Hybrid_feature(:, 2)), Num_of_FFT, Frame_size);

MSE_Hybrid_frame_10 = mean(abs(H_Hybrid_frame - RS).^2, 'all');

% Hybrid structure 20% pruning

H_Hybrid_feature = transformer.model(Feature_signal, parameters_20);

H_Hybrid_frame = reshape(extractdata(H_Hybrid_feature(:, 1)), Num_of_FFT, Frame_size) + 1j * reshape(extractdata(H_Hybrid_feature(:, 2)), Num_of_FFT, Frame_size);

MSE_Hybrid_frame_20 = mean(abs(H_Hybrid_frame - RS).^2, 'all');

% Hybrid structure 30% pruning

H_Hybrid_feature = transformer.model(Feature_signal, parameters_30);

H_Hybrid_frame = reshape(extractdata(H_Hybrid_feature(:, 1)), Num_of_FFT, Frame_size) + 1j * reshape(extractdata(H_Hybrid_feature(:, 2)), Num_of_FFT, Frame_size);

MSE_Hybrid_frame_30 = mean(abs(H_Hybrid_frame - RS).^2, 'all');

% Hybrid structure 50% pruning

H_Hybrid_feature = transformer.model(Feature_signal, parameters_50);

H_Hybrid_frame = reshape(extractdata(H_Hybrid_feature(:, 1)), Num_of_FFT, Frame_size) + 1j * reshape(extractdata(H_Hybrid_feature(:, 2)), Num_of_FFT, Frame_size);

MSE_Hybrid_frame_50 = mean(abs(H_Hybrid_frame - RS).^2, 'all');

% Hybrid structure 70% pruning

H_Hybrid_feature = transformer.model(Feature_signal, parameters_70);

H_Hybrid_frame = reshape(extractdata(H_Hybrid_feature(:, 1)), Num_of_FFT, Frame_size) + 1j * reshape(extractdata(H_Hybrid_feature(:, 2)), Num_of_FFT, Frame_size);

MSE_Hybrid_frame_70 = mean(abs(H_Hybrid_frame - RS).^2, 'all');

% LS MSE calculation in each frame
LS_MSE_in_frame(Frame, 1) = MSE_LS_frame;

% Hybrid MSE calculation in each frame

Hybrid_MSE_in_frame(Frame, 1) = MSE_Hybrid_frame;

Hybrid_MSE_in_frame_10(Frame, 1) = MSE_Hybrid_frame_10;

Hybrid_MSE_in_frame_20(Frame, 1) = MSE_Hybrid_frame_20;

Hybrid_MSE_in_frame_30(Frame, 1) = MSE_Hybrid_frame_30;

Hybrid_MSE_in_frame_50(Frame, 1) = MSE_Hybrid_frame_50;

Hybrid_MSE_in_frame_70(Frame, 1) = MSE_Hybrid_frame_70;

end

% MSE calculation

MSE_LS_over_SNR(SNR_Range == SNR, 1) = sum(LS_MSE_in_frame, 1) / Num_of_frame_each_SNR;

MSE_Hybrid_over_SNR(SNR_Range == SNR, 1) = sum(Hybrid_MSE_in_frame, 1) / Num_of_frame_each_SNR;

MSE_Hybrid_10_over_SNR(SNR_Range == SNR, 1) = sum(Hybrid_MSE_in_frame_10, 1) / Num_of_frame_each_SNR;

MSE_Hybrid_20_over_SNR(SNR_Range == SNR, 1) = sum(Hybrid_MSE_in_frame_20, 1) / Num_of_frame_each_SNR;

MSE_Hybrid_30_over_SNR(SNR_Range == SNR, 1) = sum(Hybrid_MSE_in_frame_30, 1) / Num_of_frame_each_SNR;

MSE_Hybrid_50_over_SNR(SNR_Range == SNR, 1) = sum(Hybrid_MSE_in_frame_50, 1) / Num_of_frame_each_SNR;

MSE_Hybrid_70_over_SNR(SNR_Range == SNR, 1) = sum(Hybrid_MSE_in_frame_70, 1) / Num_of_frame_each_SNR;

end

Denoising_gain = 10 * log10(MSE_LS_over_SNR ./ MSE_Hybrid_over_SNR);
Denoising_gain_10 = 10 * log10(MSE_LS_over_SNR ./ MSE_Hybrid_10_over_SNR);
Denoising_gain_20 = 10 * log10(MSE_LS_over_SNR ./ MSE_Hybrid_20_over_SNR);
Denoising_gain_30 = 10 * log10(MSE_LS_over_SNR ./ MSE_Hybrid_30_over_SNR);
Denoising_gain_50 = 10 * log10(MSE_LS_over_SNR ./ MSE_Hybrid_50_over_SNR);
Denoising_gain_70 = 10 * log10(MSE_LS_over_SNR ./ MSE_Hybrid_70_over_SNR);
