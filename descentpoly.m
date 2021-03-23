% DESCENTPOLY Execute a gradient descent process
%
% Starting at theta0, search for the minimum of the target function tf
% by iteratively applying a gradient descent process, where the
% gradient of the target function is given by gtf.  The original
% scalar input data is stored in the column vector Xo and the original
% data labels are stored in Yo, i.e. in the house areas/price linear
% regression examples Xo holds the areas and Yo the prices of the
% training data.
%
% The target function tf and its gradient must follow the interface
%
%   function loss = tf(theta,Xo,Yo) , and
%   function grad = gtf(theta,Xo,Yo)
%
% where theta is a vector holding a set of parameters, whose dimension
% specifies the polynomial order.
%
% For instance, for linear regression theta must have two
% dimensions, and if a cubic regression is desired, then theta must have
% four dimensions.  The loss function tf returns a scalar
% and the gradient gtf a vector with  the same number of dimensions
% as the vector theta.
%
% The following parameters are required:
%
% tf: target function computing the loss (or error)
% gtf: gradient of target function
% theta0: initial point for the iteration
% Xo: vector holding the original data (e.g. the house areas)
% Yo: vector holding the original outputs (e.g. the house prices)
% lr: learning rate
%
% The following parameter pairs are optional:
% "method",method: Use "batch","stochastic", "momentum", "rmsprop", "adam"
% "beta",float: beta parameter for momenum (default: 0.7)
% "beta2",float: beta2 parameter for adam (default: 0.99)
% "maxiter",int: maximum number of iterations (default: 200)
% "epsilon",float: tolerance error for convergence (default: 0.001)
% "minibatch",int: size of minibatch (default: 1)
%
% The function should return all intermediate theta values in pos, as
% well as the corresponding error en each of those positions.
%
% Example:
% [pos,errors]=descentpoly(@J,@gradJ,[0 0 0.5],X,Y,0.1,"method","adam","maxIter",10)

%
% (C) 2021 Pablo Alvarado Tarea 2, I Semestre 2021 EL5852
% Introducción al Reconocimiento de Patrones Escuela de Ingeniería
% Electrónica Tecnológico de Costa Rica
%
% (C) 2021 <Su Copyright AQUÍ>


function [thetas,errors]=descentpoly(tf,gtf,theta0,Xo,Yo,lr,varargin)

  %% Parse all given parameters
  order = length(theta0)-1;

  if (order<1)
    error("Initial point theta0 must have at least 2 dimensions");
  elseif (order>20)
    error("Currently order limit set to 20");
  endif

  ## theta0 must be a row vector
  if (isvector(theta0))
    theta0=theta0(:).';
  else
    error("theta0 must be a row vector");
  endif
    
  ## Xo must be a column vector
  if (isvector(Xo))
    Xo=Xo(:);
  else
    error("Xo must be a column vector");
  endif

  ## Yo must be a column vector
  if (isvector(Yo))
    Yo=Yo(:);
  else
    error("Yo must be a column vector");
  endif
  
  defaultMethod="batch";
  defaultBeta=0.7;
  defaultBeta2=0.99;
  defaultMaxIter=200;
  defaultEps=0.001;
  defaultMinibatch=1;

  p = inputParser;
  validMethods={"batch","stochastic","momentum","rmsprop","adam"};
  checkMethod = @(x) any(validatestring(x,validMethods));
  addParameter(p,'method',defaultMethod,checkMethod);

  checkBeta = @(x) isreal(x) && isscalar(x) && x>=0 && x<=1;
  checkRealPosScalar = @(x) isreal(x) && isscalar(x) && x>0;

  addParameter(p,'beta',defaultBeta,checkBeta);
  addParameter(p,'beta2',defaultBeta2,checkBeta);
  addParameter(p,'maxiter',defaultMaxIter,checkRealPosScalar);
  addParameter(p,'epsilon',defaultEps,checkRealPosScalar);
  addParameter(p,'minibatch',defaultMinibatch,checkRealPosScalar);
  
  parse(p,varargin{:});
  
  if ~checkBeta(lr)
    error("Learning rate must be between 0 and 1");
  endif

  ## ################################################################
  ## Your code in here!!!
  ##
  ## Next lines are just an example.  You should change them

  method = p.Results.method;       ## String with desired method
  beta = p.Results.beta;           ## Momentum parameters beta
  beta2 = p.Results.beta2;         ## ADAM paramter beta2
  maxiter = p.Results.maxiter;     ## maxinum number of iterations
  epsilon = p.Results.epsilon;     ## convergence error tolerance
  minibatch = p.Results.minibatch; ## minibatch size
  
  thetas = [theta0];
  errors = [tf(theta0,Xo,Yo)];
  
endfunction

