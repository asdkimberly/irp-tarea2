%% Loss function
%% Objective function of the parameters theta using the data and labels

function res=loss(theta,XX,Y)
  ## Calculando el residuo para las tuplas
  R=(hypothesis(XX,theta')-Y*ones(1,rows(theta)));
  ## Sumando el producto punto de los residuos en un escalar
  res=0.5*sum(R.*R,1)';
endfunction;
