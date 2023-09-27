% -----------------------------------------------------------------------
% This file is part of the ASTRA Toolbox
% 
% Copyright: 2010-2022, imec Vision Lab, University of Antwerp
%            2014-2022, CWI, Amsterdam
% License: Open Source under GPLv3
% Contact: astra@astra-toolbox.com
% Website: http://www.astra-toolbox.com/
% -----------------------------------------------------------------------

% Create a basic 256x256 square volume geometry
k=1; %experimenté con doble tamaño
vol_geom = astra_create_vol_geom(2122*k, 2122*k);

% Create a parallel beam geometry with 180 angles between 0 and pi, and
% 384 detector pixels of width 1.
% For more details on available geometries, see the online help of the
% function astra_create_proj_geom .
proj_geom = astra_create_proj_geom('parallel', 1.0, 2122*k, linspace2(0,pi,180));

P=im2double(rgb2gray(imread('d:/input/blurred1.png')));
tic

recon_id=astra_mex_data2d('create', '-vol',vol_geom,0);
cfg=astra_struct('FBP_CUDA');
cfg.ReconstructionDataId=recon_id;


[sino_id, sinogram] = astra_create_sino_gpu(P, proj_geom, vol_geom);
cfg.ProjectionDataId=sino_id;
alg_id=astra_mex_algorithm('create',cfg);
astra_mex_algorithm('run',alg_id);

reconstruction = astra_mex_data2d('get', recon_id);

toc




 figure(1); imshow(P, []);
 figure(2); imshow(sinogram, []);

imshow(reconstruction,[])
% Free memory
astra_mex_data2d('delete', alg_id);
