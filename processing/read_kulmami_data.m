%% Script to read data from KulMaMi
% Reads a .txt file created during KulMaMi-measurements and saves the
% channel matrix as a .mat-file
%
% 1) Replace "," with "."
% 2) Remove base station antennas not present in the measurements
% 3) Recombine Re- and Im-parts
% 4) Create H as [snapshots x freq points x BS antennas x users]
% 5) Rearrange M and K dimensions
% 6) If activated, remove outliers (snapshots lost, possibly due to lost sync)
% 7) Sort the base station antennas as in the physical order
% 8) If needed, remove PN sequence (old measurements)
% 9) Save the processed channel matrix in a .mat-file, or separate files
% per UE

clear all; close all;

%% Settings - set according to the current measurement
active_layers = [0:11]; % Define the active layers in the measurement
bs_antennas = 32; % Actual antennas used when measuring
file_name = 'C:\Users\Calle\Documents\MARRMOT-measurements\measurements\A-1-a\raw-channel.txt'; % File to read
save_file = 'C:\Users\Calle\Documents\MARRMOT-measurements\A-1-a\channel.mat'; % Name of .mat file where H should be saved
remove_pn = 0; % For the old measurements it needs to be compensated for the PN sequence used for the pilots
remove_out = 0; % Remove outlier (1), not remove outliers (0)

%% Variables
nbr_of_layers = length(active_layers); %2; % # of layers present in file
rsrc_blocks = 100; % # of resource blocks
block_size = 12; % Subcarriers per resource block
nbr_of_subcarriers = rsrc_blocks*block_size; % Number of subcarriers per symbol

%% Read data
A = load(file_name); % Load file
size_A = size(A);
K = nbr_of_layers; % # of users
M = bs_antennas; % 100
F = rsrc_blocks; % 100
N = size_A(1)/((nbr_of_subcarriers*2)/block_size*K); % N = number of symbols or snapshots
subcarriers = (nbr_of_subcarriers/block_size*K); % Relevant subcarriers (100/user)

%% If the following is true it means that there are commas instead of points
% Replaces all occurences of comma (",") with point (".") in the text file
% Note that the file is overwritten, which is the price for high speed
if size_A(2) > 32 % 150
    file = memmapfile(file_name, 'writable', true);
    comma = uint8(',');
    point = uint8('.');
    file.Data(transpose(file.Data == comma)) = point;
    % Update A and variables
    A = load(file_name);
    size_A = size(A);
    N = size_A(1)/((nbr_of_subcarriers*2)/block_size*K); 
    subcarriers = (nbr_of_subcarriers/block_size*K);
end

%% If more BS antennas in measurement file than actual BS antennas,
% i.e., 128 > 100, then remove the last ones
if size_A(2) > M
    A = A(:, 1:M);
    size_A = size(A); % Update size
end

%% Create complex matrix
% Recreate Re and Im for present layers (subcarriers) for each symbol (N) to a complex matrix
A_complex = zeros(size_A(1)/2, M);

for n = 1:N
    A_complex(1+(n-1)*subcarriers:n*subcarriers,:) = A((n-1)*2*subcarriers+1:(n-1)*2*subcarriers+subcarriers,:)+1i*(A((n-1)*2*subcarriers+subcarriers+1:n*2*subcarriers,:));
end

clear A; 

%% Create channel matrix 
% [snapshots x freq points x BS antennas x users]
H_temp = zeros(N,F,M,K);

for n = 0:N-1
    for f = 0:F-1
        for k = 1:K
            H_temp(n+1,f+1,:,k) = A_complex(n*subcarriers+k+f*K, :);
        end
    end
end

clear A_complex;

%% Rearrange M and K dimension
% [snapshots x freq points x BS antennas x users]
H = zeros(N,F,M,K);

a = [1 5 9 13 17 21 25 29];
ant = [repmat(a,[1 4]) repmat(a+1,[1 4]) repmat(a+2,[1 4]) repmat(a+3,[1 4])];
idx = 1;
for k = 1:4
   for m = 1:32
      layer(idx) = ceil(m/8);
      H(:,:,m,k) = H_temp(:,:,ant(idx),layer(idx));
      idx = idx + 1;
   end
end

idx = 1;
for k = 5:8
   for m = 1:M
      layer(idx) = 4+ceil(m/8);
      H(:,:,m,k) = H_temp(:,:,ant(idx),layer(idx));
      idx = idx + 1;
   end
end

idx = 1;
for k = 9:12
   for m = 1:M
      layer(idx) = 8+ceil(m/8);
      H(:,:,m,k) = H_temp(:,:,ant(idx),layer(idx));
      idx = idx + 1;
   end 
end

clear H_temp;

%% Remove outliers 
% (possibly due to lost sync)

if remove_out

    idx = 1;
    outliers = [];

    for n = 1:N
        mean_snap(n) = mean(mean(abs(H(n,:,:,1))));
        if mean_snap(n) < 0.005
            outliers(idx) = n;
            idx = idx + 1;
        end
    end

    remove_outliers = outliers; % Make it possible to see the original snapshot # of the outliers if needed

    for o = 1:length(outliers)
        H(remove_outliers(o),:,:,:) = [];
        remove_outliers = remove_outliers - 1;
    end

    N = N - length(outliers);

end

%% Sort the base station antenna vector according to physical antenna array
% After the matrix matches the physical setup

antenna_vector = [1  2  3  4  5  6  7  8  13 14 15 16 11 12 9  10 ...
                  17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32];

H = H(:,:,antenna_vector,:); % Re-organize channel matrix

%% Multply with the conjugate of the constellation for the pilot sequence
% This is needed for the old measurements
if remove_pn
    
    seq = [0 1 3 0 0 0 0 3 1 1 2 2 3 3 0 1 2 0 2 1 0 2 0 2 2 3 1 0 2 2 0 3 3 1 0 3 0 1 1 2 ...
           1 3 1 3 0 3 1 0 3 0 2 1 1 1 0 2 1 1 3 0 0 2 0 0 1 3 0 1 0 3 2 0 0 3 1 0 0 1 3 2 ...
           1 2 2 0 2 2 2 0 2 0 2 1 1 2 3 1 2 2 3 3 3 1 1 1 1 1 2 2 1 2 3 0 3 0 1 0 1 2 3 1 ...
           0 2 1 1 0 2 2 2 1 2 0 0 3 2 2 0 0 2 1 2 0 3 0 0 0 3 2 3 2 0 1 1 0 3 3 1 1 3 3 0 ...
           3 1 0 0 1 2 2 2 1 2 1 3 1 1 1 3 0 1 1 2 0 3 2 0 3 1 1 0 1 1 3 2 2 0 0 1 1 3 0 1 ...
           1 2 2 1 1 2 2 2 3 2 3 1 1 3 3 2 2 3 1 1 0 3 0 2 0 3 3 2 1 0 0 3 1 2 2 1 2 3 1 2 ...
           3 3 2 2 2 2 3 2 3 0 1 2 2 3 3 3 3 0 2 1 0 2 0 3 2 3 1 0 3 3 0 1 1 0 0 1 0 3 2 2 ...
           3 3 3 1 2 3 3 3 3 1 1 2 2 0 1 2 1 1 1 0];

    pilot_seq = [seq seq seq seq];
    present_pilots = [];
    pilot_mapping = [];
    idx = 1;

    for r = 1:rsrc_blocks
        for layer = 1:nbr_of_layers
            current_pilot = pilot_seq(r*block_size-11+active_layers(layer));
            present_pilots = [present_pilots current_pilot];
            % Constellation mapping
            if current_pilot == 0; pilot_mapping(idx) = (1/sqrt(2))*(-1-1i); end
            if current_pilot == 1; pilot_mapping(idx) = (1/sqrt(2))*(1-1i);  end
            if current_pilot == 2; pilot_mapping(idx) = (1/sqrt(2))*(-1+1i); end
            if current_pilot == 3; pilot_mapping(idx) = (1/sqrt(2))*(1+1i);  end
            idx = idx + 1;
        end
    end

    for k = 1:K
        user_pilots = pilot_mapping(k:nbr_of_layers:end);
        for f = 1:F
            %H(:,f,:,k) = H(:,f,:,k).*conj(user_pilots(f));
            Hp(:,f,:,k) = H(:,f,:,k).*user_pilots(f);
        end
    end

end

%% Save processed channel matrix in .mat-file
%save(save_file, 'H');

%% Save processed channel matrix per UE in .mat-file

for k = 1:K
    save_file_ue = [save_file '_ue' num2str(k) '.mat'];
    H_ue = H(:,:,:,k);
    save(save_file_ue, 'H_ue');
end