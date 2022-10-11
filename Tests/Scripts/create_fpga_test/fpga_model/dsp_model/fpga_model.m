function [ file_info ] = fpga_model(data_file, cfg_file, frame_num=[], tile_num=[], trace_num=[], gen_csv=1, gen_tile=1, vis=0) % TODO: trace should be something like PD or angpos and laser
% =============================================================================
%  Copyright (c) LeddarTech Inc. All rights reserved.
% =============================================================================
%  PROPRIETARY DATA NOTICE
%  No part of this file shall be used, communicated, reproduced
%  or copied in any form or by any means without the prior written
%  permission of LeddarTech Inc.
% =============================================================================
% FPGA DSP chain model
%
% fpga_model(data_file, cfg_file)
%
% data_file: Incoming data file
% cfg_file: Filename of JSON configuration
% frame_num: Frame number to process (list or range)
% tile_num: Tile number to process (list or range)
% trace_num: Trace number to process (list or range)
% gen_csv: Generate modules CSV files
% gen_tile: Generate tile output file
% vis: Enable visualization when not 0
%

% this will have to be removed with Octave 7
% jsonencode and jsondecode will be implemented (probably in another package...)
try
  pkg load jsonstuff;
  jsondecode('{"test": 0}');
catch
  if( sscanf(ver('Octave').Version, '%f.%u')(1) < 7.0 )
    display("Installing jsonstuff")
    pkg install https://github.com/apjanke/octave-jsonstuff/releases/download/v0.3.3/jsonstuff-0.3.3.tar.gz;
    pkg load jsonstuff;
  else
    error('Octave 7.0 detected but jsondecode not found; It is supposed to be present.  Probably missing package; fix script.\n');
  endif
end_try_catch

  file_info = {};
  cfg = jsondecode(fileread(cfg_file));
  tile_cfg = jsondecode(fileread("tile_cfg.json"));


  if ischar(data_file)
    [dir, name, ext] = fileparts(data_file);
    switch ext
      case '.txt'
        data = get_raw_data(data_file, {});
      case '.dat'
        data = read_dat(data_file);
      case '.csv'
        data = read_csv(data_file,';');
      otherwise
        error('File extension unknown.');
    endswitch
  else
    if iscell(data_file)
      ext = '.txt';
    else
      ext = '.dat';
    endif
    data = data_file;
  endif

  nfo.frame_number = 0;
  nfo.pd_number = 0;
  nfo.angpos_laser = 0;
  nfo.cfg_id      = 0; % dummy until properly mapped
  nfo.frame_id    = 0; % dummy until properly mapped
  nfo.optical_id  = 0; % dummy until properly mapped
  nfo.acq_id      = 0; % dummy until properly mapped

  %Add new config from tile_cfg.json if not already in cfg (to avoid overwrite in batch sim)
  if (isfield(cfg.sn_rem,'set_sel') == 0)
     cfg.sn_rem.set_sel = tile_cfg.sn_rem.set_sel;
     cfg.sn_rem.block_enable = tile_cfg.sn_rem.block_enable;
     cfg.sn_rem.template = tile_cfg.sn_rem.template;
     cfg.sn_rem.global_offset = tile_cfg.sn_rem.global_offset;
     cfg.blinder.block_enable = tile_cfg.blinder.block_enable;
     cfg.mfilter.block_enable = tile_cfg.mfilter.block_enable;
  endif

  if (isfield(cfg.peak_detect,'sample_select_ctrl') == 0)
     cfg.peak_detect.sample_select_ctrl = tile_cfg.peak_detect.sample_select_ctrl;
  endif

% data is a complete tile or frame (.txt files)
  if( ext == '.txt' && iscell(data) )

    s = size(data);
    if( ~isempty(frame_num) )
      frame_range = frame_num+1;
    else
      frame_range = [1:s(1)];
    endif

    if( ~isempty(tile_num) )
      tile_range = tile_num+1;
    else
      tile_range = 1:s(2);
    endif

    first = 1;
    for frame=frame_range
      for tile=tile_range;
        if( isfield(data{frame, tile}, 'raw') )
          s = size(data{frame, tile}.raw);
          nfo.nb_samples = data{frame, tile}.cfg.BasePoints;
          if( nfo.nb_samples~=s(1) )
            fprintf('Mismatch between data and configuration for frame %u, tile %u; cfg: %u, real: %u - discarding.', frame-1, tile-1, nfo.nb_samples, s(1));
          else
            if( ~isempty(trace_num) )
              trace_range = trace_num+1;
            else
              trace_range = [1:s(2)];
            endif

            for j=trace_range
              nfo.timestamp = time;
              nfo.frame_number = frame;
	          for i=1:4
	             if(tile_cfg.tile.pen(i) == 1)
	                partition = i-1;
	                break;
	             endif
	          end

              nfo.pd_number = partition;

              file_info = dsp_chain(data{frame, tile}.raw(:, j), nfo, cfg, j-1, first, gen_csv, gen_tile, vis);

              nfo.angpos_laser = nfo.angpos_laser + 1;

		      if(nfo.angpos_laser == (tile_cfg.tile.nap*tile_cfg.tile.nangbk))
		         nfo.pd_number = nfo.pd_number + 4;
		         nfo.angpos_laser = 0;
		         if(nfo.pd_number >= 64)
		            next_partition_found = 0;
		            while(next_partition_found == 0)
		               if(partition+1 > 3)
		                  partition = 0;
		               else
		                  partition+=1;
		               endif
		               if(tile_cfg.tile.pen(partition+1) == 1)
		                  next_partition_found = 1;
		               endif
		            end
		            nfo.pd_number = partition;
		         endif
		      endif
              first = 0;
            endfor
          endif
        endif
      endfor
    endfor

  else

    s = size(data);
    nfo.nb_samples = s(1);

    if( ~isempty(trace_num) )
      trace_range = trace_num;
    else
      trace_range = [1:s(2)];
    endif

    for i=1:4
       if(tile_cfg.tile.pen(i) == 1)
          partition = i-1;
          break;
       endif
    end

    nfo.pd_number = partition;

    first = 1;
    for j=trace_range
      display(j)
      if iscell(data)
        d = data{j}(:);
      else
        d = data(:, j);
      endif

      nfo.timestamp = time;

      dsp_chain(d, nfo, cfg, j-1, first, gen_csv, gen_tile, vis);

      nfo.angpos_laser = nfo.angpos_laser + 1;

      if(nfo.angpos_laser == (tile_cfg.tile.nap*tile_cfg.tile.nangbk))
         nfo.pd_number = nfo.pd_number + 4;
         nfo.angpos_laser = 0;
         if(nfo.pd_number >= 64)
            next_partition_found = 0;
            while(next_partition_found == 0)
               if(partition+1 > 3)
                  partition = 0;
               else
                  partition+=1;
               endif
               if(tile_cfg.tile.pen(partition+1) == 1)
                  next_partition_found = 1;
               endif
            end
            nfo.pd_number = partition;
         endif
      endif
      first = 0;
    endfor

  endif

endfunction

% vim:tw=0:ts=2:sts=2:sw=2:et
