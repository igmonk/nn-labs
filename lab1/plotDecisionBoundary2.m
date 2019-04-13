function plotDecisionBoundary2(theta, X, y)

  % Plot Data
  plotData2(X(:,2:3), y);
  hold on

  if size(X, 2) <= 3
    % Only need 2 points to define a line, so choose two endpoints
    plot_x = [min(X(:,2))-2,  max(X(:,2))+2];

    % Calculate the decision boundary line
    plot_y1 = (-1./theta(3)).*(theta(2).*plot_x + theta(1));
    plot_y2 = (-1./theta(3,2)).*(theta(2,2).*plot_x + theta(1,2));
    plot_y3 = (-1./theta(3,3)).*(theta(2,3).*plot_x + theta(1,3));
    plot_y4 = (-1./theta(3,4)).*(theta(2,4).*plot_x + theta(1,4));

    % Plot, and adjust axes for better viewing
    plot(plot_x, plot_y1);
    plot(plot_x, plot_y2);
    plot(plot_x, plot_y3);
    plot(plot_x, plot_y4);

    % Legend, specific for the exercise
    legend('Class 1', 'Class 2', 'Class 3', 'Class 4', 'Decision Boundary');
    axis([0, 100, 0, 100]);
  else
    % Here is the grid range
    u = linspace(-1, 1.5, 50);
    v = linspace(-1, 1.5, 50);

    z = zeros(length(u), length(v));
    % Evaluate z = theta*x over the grid
    for i = 1:length(u)
      for j = 1:length(v)
      z(i,j) = mapFeature(u(i), v(j))*theta;
      end
    end
    z = z'; % important to transpose z before calling contour

    % Plot z = 0
    % Notice you need to specify the range [0, 0]
    contour(u, v, z, [0, 0], 'LineWidth', 2)
  end
  hold off

end
