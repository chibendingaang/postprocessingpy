program vectorized_code
    implicit none
    integer :: L, dtsymb, epss, choice, t_smoothness
    real :: dt, Lambda, Mu, epsilon
    integer :: begin, end, fine_res, steps, stepcount, ti, x
    real, dimension(:,:), allocatable :: Cxt, CExt
    real, dimension(:,:,:), allocatable :: Sp_a, Spnbr_a, E_loc, Sp_ngva, Spnbr_ngva, Eeta_loc, mconsv
    real, dimension(:), allocatable :: eta_
    real :: alpha
    character(len=50) :: dtstr, epstr, paramstm, path, alphastr
    integer :: conf
    real :: start

    call get_command_argument(1, L)
    call get_command_argument(2, dtsymb)
    call get_command_argument(3, Lambda)
    call get_command_argument(4, Mu)
    call get_command_argument(5, begin)
    call get_command_argument(6, end)
    call get_command_argument(7, epss)
    call get_command_argument(8, choice)
    call get_command_argument(9, t_smoothness)

    dt = 0.001 * dtsymb
    epsilon = 10.0 ** (-epss)
    
    if (dtsymb == 2) then
        fine_res = 1 * t_smoothness
    else if (dtsymb == 1) then
        fine_res = 2 * t_smoothness
    endif

    write(dtstr, '(I1,A)') dtsymb, 'emin3'
    write(epstr, '(I1,A)') epss, 'eps_min3'

    if (Lambda == 1 .and. Mu == 0) then
        paramstm = 'hsbg'
    else if (Lambda == 0 .and. Mu == 1) then
        paramstm = 'drvn'
    else
        paramstm = 'a2b0'
    endif

    alpha = (Lambda - Mu) / (Lambda + Mu)
    write(alphastr, '(F6.2)') alpha

    if (choice == 0) then
        param = 'xp' // paramstm
        if (paramstm /= 'a2b0') then
            write(path, '(A,I0,A)') './', param, '/L', L, '/2emin3'
        else
            write(path, '(A,I0,A,A,A)') './', param, '/L', L, '/alpha_', alphastr, '/2emin3'
        endif
    else if (choice == 1) then
        param = 'qw' // paramstm
        if (paramstm /= 'a2b0') then
            write(path, '(A,I0,A)') './', param, '/L', L, '/2emin3'
        else
            param = 'qwa2b0'
            write(path, '(A,A,A,A,A)') './', param, '/2emin3/alpha_', alphastr
        endif
    else if (choice == 2) then
        if (paramstm == 'a2b0') then
            param = 'qw' // paramstm
            write(path, '(A,A,A,A,A)') './', param, '/2emin3/alpha_', alphastr
        else
            param = paramstm
            write(path, '(A,I0,A)') './', param, '/L', L, '/2emin3'
        endif
    endif

    start = cpu_time()

    do conf = begin, end - 1
        call obtaincorrxt(conf, path, Cxt, Sp_a, stepcount, fine_res, dt, L, alpha, param)
        if (param == 'xpa2b0' .or. param == 'qwa2b0') then
            call save_npy('./Cxt_series_storage/L' // L // '/alpha_ne_pm1/Cxt_t_' // dtstr // '_jump' // fine_res // '_' // epstr // '_' // param // '_' // alphastr // '_' // conf // 'to' // conf + 1 // 'config.npy', Cxt)
        endif
    enddo

contains

    subroutine obtaincorrxt(file_j, path, Cxt, Sp_a, stepcount, fine_res, dt, L, alpha, param)
        integer, intent(in) :: file_j, stepcount, fine_res, L
        real, intent(in) :: dt, alpha
        character(len=*), intent(in) :: path, param
        real, dimension(:,:), allocatable, intent(out) :: Cxt
        real, dimension(:,:,:), allocatable, intent(out) :: Sp_a
        integer :: steps, ti, x

        ! Read data and process as needed

        steps = 512  ! Assuming a fixed number of steps for simplicity
        allocate(Sp_a(steps, L, 3))
        allocate(Cxt(steps / fine_res + 1, L))

        ! Initialize and compute correlation

        Cxt = calc_Cxt(Sp_a, steps, fine_res, L, alpha, param)

    end subroutine obtaincorrxt

    function calc_Cxt(Sp_a, steps, fine_res, L, alpha, param) result(Cxt)
        real, dimension(:,:,:), intent(in) :: Sp_a
        integer, intent(in) :: steps, fine_res, L
        real, intent(in) :: alpha
        character(len=*), intent(in) :: param
        real, dimension(steps / fine_res + 1, L) :: Cxt
        integer :: ti, x

        ! Compute correlation

        do ti = 0, steps / fine_res
            do x = 0, L - 1
                Cxt(ti, x) = sum(Sp_a(:, x, :) * Sp_a(:, x, :) ) / (steps / fine_res + 1 - ti)
            enddo
        enddo

    end function calc_Cxt

    subroutine save_npy(filename, Cxt)
        character(len=*), intent(in) :: filename
        real, dimension(:,:), intent(in) :: Cxt

        ! Code to save array to file in .npy format

    end subroutine save_npy

end program vectorized_code
