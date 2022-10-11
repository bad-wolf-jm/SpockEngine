function [detections, leftovers, file_info] = peak_detection(data, threshold, sat_pulse, sideinfo, cfg)
% PEAK_DETECTION  Peak detection and selection algorithm
%
% [peaks, file_info] = peak_detection(data, cfg)
%
% IN
% data      : CFAR data
% threshold : CFAR thresholds (in-phase with data)
% sat_pulse : Saturated flag (in-phase with data)
% bl_before : Baseline before saturation
% bl_after  : Baseline after saturation
% cfg       : Configuration structure
%
% OUT
% detections   : Detections structure
% leftovers    : Number of detections not included in structure because of cfg.sample_select_ctrl.n_detections
% file_info    : File information for results writing
%


   % Check input size
   assert(size(data) == size(threshold), "Size of data & threshold don't match")
   assert(size(data,1) == 1 || size(data,2) == 1)

   if size(data,2) ~= 1
      data = data';
      threshold = threshold';
   end


   % Check for required configuration parameters
   params = { ...
      "qint_en"       ,...
      "cfar_ignore"   ,...
      "mask_length"   ,...
      "margin_start"  ,...
      "margin_end"    ,...
   };

   err = 0;
   for i = 1:length(params)
      name = params{i};
      if ~isfield(cfg.detection_ctrl, name)
         printf("Missing field cfg.%s\n", name);
         err++;
      end
   end
   assert(err == 0, "Missing required cfg field(s): %d", err)


   file_info.header  = {'trace'; 'det_index'; 'det_bin'; 'det_mag'; 'last'};
   file_info.data    = [];

   %------------------------------------------------------------------------------------------------

   % Mask data per threshold
   is_above_thld = zeros(size(data));
   is_above_thld(data >= threshold) = 1;

   % Mask data per margin
   is_within_margins = ones(size(data));

   if cfg.detection_ctrl.margin_start > 0
      is_within_margins(1:min(end, cfg.detection_ctrl.margin_start)) = 0;
   end

   if cfg.detection_ctrl.margin_end > 0
      is_within_margins(max(1, end-cfg.detection_ctrl.margin_end+1):end) = 0;
   end


   % Find peaks where a peak is defined as: x(n-1) < x(n) <= x(n+1)
   % Note: outside of the trace, -INF is assumed
   prev = zeros(size(data));
   prev(diff([-inf; data]) > 0) = 1;

   next = zeros(size(data));
   next(flip(diff(flip([data; -inf]))) >= 0) = 1;

   is_peak = prev & next;

   % Select peak candidates
   valid = is_peak & is_within_margins;

   if (cfg.detection_ctrl.cfar_ignore == 0)
      valid = valid & is_above_thld;
   end

   candidates = find(valid==1);


   % find(candidates==1)
   % data(find(candidates==1))

   %------------------------------------------------------------------------------------------------

   % No detections, quit early
   if length(candidates) == 0
      detections  = [];
      leftovers   = 0;
      return
   end


   %  Latch sideinfo on rising edge (red) of sat_pulse
   si = [];
   red = find(diff([0;sideinfo.sat_pulse])==1);
   for i=1:length(red)
      pos = red(i);
      si(i).bl_before   = sideinfo.bl_before(pos);
      si(i).bl_after    = sideinfo.bl_after(pos);
      si(i).pulse_len   = sideinfo.pulse_len(pos);
      si(i).smpl_before = sideinfo.smpl_before(pos);
   end


   % Populate the detections
   for i=1 : length(candidates)
      pos = candidates(i);

      % Initialize the detection structure
      detections(i).idx  = pos-1; % impl first sample is 0 not 1
      detections(i).bin  = detections(i).idx;
      detections(i).mag  = data(pos);
      detections(i).bl_before    = 0;
      detections(i).bl_after     = 0;
      detections(i).pulse_len    = 0;
      detections(i).smpl_before  = 0;
      detections(i).saturated    = 0;

      % Apply correction/interpolation
      if sat_pulse(pos) == 1
         detections(i).saturated = 1;

         % Use latched side information
         assert(numel(si) > 0, "Not enough latched sideinfo")
         detections(i).bl_before    = si(1).bl_before;
         detections(i).bl_after     = si(1).bl_after;
         detections(i).pulse_len    = si(1).pulse_len;
         detections(i).smpl_before  = si(1).smpl_before;

         si(1) = []; % discard used sideifno

      elseif cfg.detection_ctrl.qint_en ~= 0
         % Skip qint on the edges of the trace
         if pos > 1 && pos < length(data)
            x0 = detections(i).idx;
            y  = [data(pos-1), data(pos), data(pos+1)];
            [detections(i).bin detections(i).mag] = quadratic_interpolation(x0, y);
         end
      end
   end

   %------------------------------------------------------------------------------------------------

   % Selection and masking
   select_cnt  = 0;
   while length(detections) && select_cnt < cfg.sample_select_ctrl.n_detections
      % Select the next detection among the remaining ones
      select_cnt++;
      selected(select_cnt) = detections(1);

      % Remove detection within the mask region
      lo = detections(1).bin - cfg.detection_ctrl.mask_length;
      hi = detections(1).bin + cfg.detection_ctrl.mask_length;
      bins = [detections(:).bin];
      detections = detections(find(bins < lo | bins > hi));
   end

   %------------------------------------------------------------------------------------------------
   fmt = peak_detection_format();

   % Outputs

   leftovers = length(detections);

   if exist('selected')
      detections = selected;
      file_info.data = [ [detections(:).idx]', [detections(:).bin]' .* 2**fmt.bin.fractional, [detections(:).mag]' ];
   else
      detections = [];
      file_info.data = [];
   end

endfunction


function sfxpt= to_sfxpt(float, integer, fractional, rounding="truncation")
% TO_SFXPT Convert a float to a signed fixed point number:
%     - integer: number of bits for the integer part, including sign bit
%     - fractional: number of bits for fractional part
%     - total bit-width of the number is : integer + fractional
%     - saturate on overflow

   assert(isa(float, "float"))
   assert(integer > 0)
   assert(fractional >= 0)

   if float > 2**(integer-1) - 2**-fractional
      % Saturate on overflow
      fprintf("to_sfxpt(%.6f, %d, %d): overflow\n", float, integer, fractional)
      sfxpt = 2**(integer-1) - 2**-fractional;

   elseif float < -2**(integer-1)
      % Saturate on underflow
      fprintf("to_sfxpt(%.6f, %d, %d): underflow\n", float, integer, fractional)
      sfxpt = -2**(integer-1);

   else
      switch (rounding)
         case 'truncation'
            sfxpt = floor(float * 2**fractional) * 2**-fractional;
         case 'convergent'
            sfxpt = roundb(float * 2**fractional) * 2**-fractional;
         otherwise
            error("Unknonw rounding method: %s", rounding);
      endswitch
   end
endfunction


function ufxpt= to_ufxpt(float, integer, fractional, rounding="truncation")
   % TO_UFXPT Convert a float to an unsigned fixed point number:
   %     - integer: number of bits for the integer part, including sign bit
   %     - fractional: number of bits for fractional part
   %     - total bit-width of the number is : integer + fractional
   %     - saturate on overflow


   assert(isa(float, "float"))
   assert(integer > 0)
   assert(fractional >= 0)


   if float > 2**(integer) - 2**-fractional
      % Saturate on overflow
      fprintf("to_ufxpt(%.6f, %d, %d): overflow\n", float, integer, fractional)
      ufxpt = 2**(integer-1) - 2**-fractional;

   elseif float < 0
      % Saturate on underflow
      fprintf("to_ufxpt(%.6f, %d, %d): underflow\n", float, integer, fractional)
      ufxpt = 0;

   else
      switch (rounding)
         case 'truncation'
            ufxpt = floor(float * 2**fractional) * 2**-fractional;
         case 'convergent'
            ufxpt = roundb(float * 2**fractional) * 2**-fractional;
         otherwise
            error("Unknonw rounding method: %s", rounding);
      endswitch
   end
endfunction


function [xi, yi] = quadratic_interpolation(x0,y)
   % QUADRATIC_INTERPOLATION  Quadratic interpolation of three adjacent/equispaced samples fitted on a parabola
   %  given by: y(x) = a*(x-p)^2 + b
   %
   %  x0 : position of y[0]
   %  y  : array with 3 amplitudes (ie: y[-1], y[0], y[1])
   %  xi : estimated peak position
   %  yi : estiamted peak amplitude

   assert(isscalar(x0))
   assert(length(y) == 3)
   assert((y(2) > y(1) && y(2) >= y(3)) || (y(2) < y(1) && y(2) <= y(3)), "Expecting a peak")


   % peak at (x[0],y[0])
   % p   = 0.5 * (y[1]-y[-1]) / (2*y[0]-y[-1]-y[1])
   % xi  = x[0] + p
   % yi  = y[0] - 0.25 * p * (y[-1]-y[1])

   p = 0.5 * (y(3) - y(1)) / (2*y(2) - y(1) - y(3));
   xi = x0 + p;
   yi = y(2) - 0.25 * p * (y(1) - y(3));

   xi = to_ufxpt(xi, 10, 6, 'convergent');
   yi = to_sfxpt(yi, 16, 0, 'convergent');
endfunction


function err_cnt = test_qint()
% TEST_QINT Minimal pseudo unit test for quadratic_interpolation

   err_cnt = 0;

   % fmt: x0, [y(-1), y(0), y(1)], exp_x, exp_y
   test_data = [ ...
      [  248  15240  14894   17194   247.625000   14714.0 ]; ...
      [  248  17194  18960   18746   248.390625   19112.0 ]; ...
      [ 1000      1   8192  -32768   999.671875   10923.0 ]; ...
      [  100   1899   8191    8191   100.500000    8978.0 ]; ...
      [  100   3965   4096    1105    99.546875    4423.0 ]; ...
      [   23   2805   2815    2799    22.890625    2815.0 ]; ...
      [   59  10477  10865    7691    58.609375   11137.0 ]; ...
      [  248  15240  14894   17194   247.625000   14714.0 ]; ...
      [  250  17194  18960   18746   250.390625   19112.0 ]; ...
      [  454  -1023   2048    -500   454.046875    2054.0 ]; ...
      [  454  -1023   4096    -500   454.031250    4100.0 ]; ...
      [  454  -1023   8192    -500   454.015625    8194.0 ]; ...
      [    2      4      1      10     1.750000       1.0 ]; ...
      [    2   -258      0   -1245     1.671875      81.0 ]; ...
      [    2    258      0     990     1.703125     -54.0 ]; ...
      [    2    258      0       0     2.500000     -32.0 ]; ...
      [    2   -258      0       0     2.500000      32.0 ]; ...
      [  990   4096   8192  -32768   989.593750   11962.0 ]; ...
      [  991   4096  16384  -32768   990.703125   19149.0 ]; ...
      [  992   8192  16384  -32768   991.640625   20041.0 ]; ...
      [  993   4096  16384  -16384   992.765625   17548.0 ]; ...
      [ 1022    -20   8192  -32768  1021.671875   10918.0 ]; ...
      [ 1022   -450   8192  -32768  1021.671875   10824.0 ]; ...
      [ 1022   -500   4192  -32768  1021.609375    7317.0 ]; ...
      [ 1022   -500   8192  -32768  1021.671875   10813.0 ]; ...
      [ 1022   -500  16192  -32768  1021.750000   18174.0 ]; ...
      [ 1022   -500      0  -32768  1021.515625    3912.0 ]; ...
      [ 1022   -222   8192  -32546  1021.671875   10849.0 ]; ...
      [ 1022   -223   8192  -32546  1021.671875   10849.0 ]; ...
      [ 1022   -222   8193  -32547  1021.671875   10850.0 ]; ...
      [ 1022   -222   8192  -32545  1021.671875   10849.0 ]; ...
      [ 1022   -500   8192  -32768  1021.671875   10813.0 ]; ...
      [ 1022   -501   8192  -32768  1021.671875   10813.0 ]; ...
      [ 1022   -502   8192  -32768  1021.671875   10813.0 ]; ...
      [ 1022   -503   8192  -32768  1021.671875   10813.0 ]; ...
      [ 1022   -504   8192  -32768  1021.671875   10812.0 ]; ...
      % overflow at 16-bits
      [ 1000  32766  32767  -32768  999.5000000   32767.0 ]; ...
   ];


   for n = 1 : size(test_data,1)
      curr  = test_data(n,:);

      data.x0  = curr(1);
      data.y   = curr(2:4);
      data.xi  = curr(5);
      data.yi  = curr(6);

      [xi, yi] = quadratic_interpolation(data.x0, data.y);

      if xi ~= data.xi || yi ~= data.yi
         fprintf("n=%d; quadratic_interpolation(%d, [%d %d %d])\n", n, data.x0, data.y(:))
         fprintf("got: (%.6f, %.1f)\n", xi, yi)
         fprintf("exp: (%.6f, %.1f)\n", data.xi, data.yi)
         err_cnt++;
      end
   end

   assert(err_cnt==0, "found some error in quadratic_interpolation()")
endfunction

% vim:tw=0:ts=3:sts=3:sw=3:et
