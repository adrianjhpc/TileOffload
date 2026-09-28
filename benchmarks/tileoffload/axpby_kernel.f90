module tileoff_axpby_kernel
contains

  subroutine axpby_prepare(a, b, c)
    real :: a(:), b(:), c(:)

    !$tileoff update device(a)
    !$tileoff update device(b)
    !$tileoff update device(c)
  end subroutine

  subroutine axpby_compute(alpha, beta, a, b, c)
    real :: alpha, beta
    real :: a(:), b(:), c(:)
    integer :: i

    !$tileoff parallel tile(128) pack(a:device, b:device, c:device)
    do i = 1, size(c)
      c(i) = alpha * a(i) + beta * b(i)
    end do
  end subroutine

  subroutine axpby_fetch(c)
    real :: c(:)

    !$tileoff update host(c)
  end subroutine

  subroutine axpby_release(a, b, c)
    real :: a(:), b(:), c(:)

    !$tileoff release(a, b, c)
  end subroutine

end module

