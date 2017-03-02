%GaussianProcessRegression
% Gaussian Process Regression

% Generate random data
data = -0.99 + 8 * rand(120,2);
rows = size(data, 1);
cols = size(data, 2);
% Noise parameter
beta = 11;
identity = eye(rows, rows);
gram_matrix = zeros(rows, rows);
c = 1/beta;
output = zeros(rows, 1);

% Create the Gram Matrix
for i=1:rows
    for j=1:rows
        gram_matrix(i,j) = abs(data(j,1) - data(i,1));
    end
end

% Determine covariance matrix
C = gram_matrix + (1/beta) * identity;

% Get data for regression curve
max_val = 8;
start_val = -0.99;
increment = 0.01;
test_input = start_val: increment: max_val;
iteration = ((max_val - start_val) / increment ) + 1;
pred_output = zeros(iteration, 1);

% Prediction
for i = 1:iteration
    kernel = zeros(rows, 1);
    input_data = test_input(i);
    for j = 1: rows
        % Define RBF kernel
        kernel(j, 1) = abs(data(j, 1) - input_data);
    end
    % Predict output
    pred_output(i) = kernel' * inv(C) * data(:, 2);
end

plot(data(:, 1), data(:, 2), 'rX', 'LineWidth', 1);
hold on;
plot(test_input, pred_output, 'k', 'LineWidth', 1);
title('Gaussian Process Regression');
