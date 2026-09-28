module tileoff_saxpy_kernel
contains

  subroutine saxpy_prepare(x, y)
    real :: x(:), y(:)

    !$tileoff update device(x)
    !$tileoff update device(y)
  end subroutine

  subroutine saxpy_compute(alpha, x, y)
    real :: alpha
    real :: x(:), y(:)
    integer :: i

    !$tileoff parallel tile(128) pack(x:device, y:device)
    do i = 1, size(y)
      y(i) = alpha * x(i) + y(i)
    end do
  end subroutine

  subroutine saxpy_fetch(y)
    real :: y(:)

    !$tileoff update host(y)
  end subroutine

  subroutine saxpy_release(x, y)
    real :: x(:), y(:)

    !$tileoff release(x, y)
  end subroutine

end module

