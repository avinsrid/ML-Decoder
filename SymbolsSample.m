% ; is omitted after certain lines if printing the value in the command
% line is mandatory
clear all; clf
M=4; % M-QAM Modulation
N = 10 ; % Number of symbols to transmit, for test we use N = 10
Xin = randint(1,N,M) % Our input set
y=modulate(modem.qammod(M),Xin) % Input set is M-QAM modulated

% Generating channel gain values between t1r1, t1r2, t2r1, t2r2
h1 = 1/sqrt(2)*(randn(1,1) + 1i*randn(1,1));
h2 = 1/sqrt(2)*(randn(1,1) + 1i*randn(1,1));
g1 = 1/sqrt(2)*(randn(1,1) + 1i*randn(1,1));
g2 = 1/sqrt(2)*(randn(1,1) + 1i*randn(1,1));

% Randon Noise Generation
N11=1/sqrt(2)*(randn(1,1)+1i*randn(1,1));
N12=1/sqrt(2)*(randn(1,1)+1i*randn(1,1));

% Obtaining random symbols x1, x2, x3, x4 from the modulated input set
x1 = y(randi(numel(y)));
x2 = y(randi(numel(y)));
x3 = y(randi(numel(y)));
x4 = y(randi(numel(y)));

% Alamouti block codes of input set
ala1 = [x1 -conj(x2); x2 conj(x1)];
ala2 = [x3 -conj(x4); x4 conj(x3)];

% Channel matrices
H = [h1 h2];
G = [g1 g2];
a = 1/sqrt(7);
Shaping = [1-i 1-2i;1+2i -1-i];
V = a*Shaping;
H_bar = H*V;
G_bar = G*V;
% R1 is the received signal on receiver 1 in format [r11 r12]
R1 = H*ala1 + H_bar*ala2+[N11 N12];

% R2 is the received signal on receiver 2 in format [r21 r22]
R2 = G*ala1 + G_bar*ala2+[N11 N12];

%{ This piece of code below I believe is incorrect as it will create:
%[ r11 r12 ]
%[ r21 r22 ]
% We need the matrix R in format R = [r11 + r12; r21 + r22]
R = [R1(1,1) + R1(1,2); R2(1,1) + R2(1,2)]
P = transpose([H ;G])
Q = transpose([H_bar ;G_bar]);
S = transpose([x3 x4])

% Test elements for one iteration of conditional optimization logic
s1 = [1+i; -1+i];
s2 = [1-i; -1+i];

%cbar calculation. P_adjoint is calculating the adjoint of matrix P.
%adjoint(P) did not work, hence manually calculating it.
% ref: http://math.tutorvista.com/algebra/cofactor-matrix.html
norm_f = (norm(H,'fro'))^2+(norm(G,'fro'))^2;
P_ad = transpose([P(2,2) -P(2,1); -P(1,2) P(1,1)])
C1 = (2*(P_ad)/(norm_f)) * (R-(Q*s1))
C2 = (2*(P_ad)/(norm_f)) * (R-(Q*s2))

%Square Metric for symbols 1 and 2. Which ever is lower is the more
%accurate result.
Ms_1 = (norm(R - P*C1 - Q*s1))^2
Ms_2 = (norm(R - P*C2 - Q*s2))^2