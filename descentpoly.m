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
% Introducci贸n al Reconocimiento de Patrones Escuela de Ingenier铆a
% Electr贸nica Tecnol贸gico de Costa Rica
%
% (C) 2021 Andr閟 Jim閚ez Jim閚ez y Kimberly  Orozco Retana  Tarea 2, I Semestre 2021 EL5852
% Introducci贸n al Reconocimiento de Patrones Escuela de Ingenier铆a
% Electr贸nica Tecnol贸gico de Costa Rica

%utilizando c骴igo del curso con modificaciones propias
%Sean autores= Andr閟 Jim閚ez Jim閚ez y Kimberly Orozco Retana


pkg load optim;

## Data stored each sample in a row, where the last row is the label
D=load("escazu40.dat");

## Construir la matriz de pre dise駉, o sea las columans con las 3 entradas: areas, pisos y cuartos; autores
%Xo=[D(:,1),D(:,2),D(:,3)]; %para 3D
Xo=[D(:,1)]; %solo las areas

## The outputs vector with the original data (Etiqutas)
Yo=D(:,4);



[thetas,errors]=descentpoly(@J,@gradJ,[0 0 0.5],Xo,Yo,0.1,"method","adam","maxiter",10)


function [thetas,errors]=descentpoly(tf,gtf,theta0,Xo,Yo,lr,varargin)

  %% Parse all given parameters
  order = length(theta0)-1;
  
  
  ## normalizer_type="normal";
  normalizer_type="normal";

## Normalize the data
  nx = normalizer(normalizer_type);  
  x = nx.fit_transform(Xo);

  if (order<1)
    error("El punto inicial de theta0 debe tener al menos 2 dimensiones");
  %%Creamos las matrices de dise駉 completas; autores
  %para orden 1
  elseif (order=1)
    XX=[ones(rows(x),1) x];
    
  %para orden 2  
  elseif (order=2)
    %XX=[ones(rows(x),1) x x.^2 x(:,1).*x(:,2) x(:,1).*x(:,3) x(:,2).*x(:,3)]; %Para 3D
    XX=[ones(rows(x),1) x x.^2];
    
  %para orden 3  
  elseif (order=3)
    %XX=[ones(rows(x),1) x x.^2 x(:,1).*x(:,2) x(:,1).*x(:,3) x(:,2).*x(:,3) x.^3 (x(:,1).^2).*x(:,2) (x(:,1).^2).*x(:,3) x(:,1).*(x(:,2).^2) x(:,1).*(x(:,3).^2) (x(:,2).^2).*x(:,3) x(:,2).*(x(:,3).^2) x(:,1).*x(:,2).*x(:,3)];
    XX=[ones(rows(x),1) x x.^2 x.^3];
    
  %para orden 4  
  elseif (order=4)
    %XX=[ones(rows(x),1) x x.^2 x(:,1).*x(:,2) x(:,1).*x(:,3) x(:,2).*x(:,3) x.^3 (x(:,1).^2).*x(:,2) (x(:,1).^2).*x(:,3) x(:,1).*(x(:,2).^2) x(:,1).*(x(:,3).^2) (x(:,2).^2).*x(:,3) x(:,2).*(x(:,3).^2) x(:,1).*x(:,2).*x(:,3) x.^4 (x(:,1).^3).*x(:,2) (x(:,1).^3).*x(:,3) x(:,1).*(x(:,2).^3) x(:,1).*(x(:,3).^3) (x(:,2).^3).*x(:,3) x(:,2).*(x(:,3).^3) (x(:,1).^2).*(x(:,2).^2) (x(:,1).^2).*(x(:,3).^2) (x(:,2).^2).*(x(:,3).^2) (x(:,1).^2).*x(:,2).*x(:,3) x(:,1).*(x(:,2).^2).*x(:,3) x(:,1).*x(:,2).*(x(:,3).^2)];
    XX=[ones(rows(x),1) x x.^2 x.^3 x.^4];
    
  elseif (order>5)
    error("El l韒ite de orden es de 4");
  endif

  ## theta0 must be a row vector
  if (isvector(theta0))
    theta0=theta0(:).';
  else
    error("theta0 must be a row vector");
  endif
    
  ## Xo must be a column vector No aplica para 3D
%  if (isvector(Xo))
%    Xo=Xo(:);
%  else
%   error("Xo must be a column vector");
%  endif

  ## Yo must be a column vector
  if (isvector(Yo))
    Yo=Yo(:);
  else
    error("Yo must be a column vector");
  endif
  
  J = tf(theta,Xo,Yo)
  gradJ=gradloss(XX,theta0);
  
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
  
endfunction;

