function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)

    % Initialization
    m = length(y); % number of training examples
    J_history = zeros(num_iters, 1);

    for iter = 1:num_iters

        k = (alpha / m);
        h = theta' * X';
        diff = h' - y; %' transpose(h) minus y

        temp0 = theta(1) - k * sum(sum(diff));
        temp1 = theta(2) - k * sum(sum(diff .* X(:,2)));
        temp2 = theta(3) - k * sum(sum(diff .* X(:,3)));
        theta = [temp0; temp1; temp2];

        % Save the cost J in every iteration
        J_history(iter) = costFunction(theta, X, y);

    end

end
