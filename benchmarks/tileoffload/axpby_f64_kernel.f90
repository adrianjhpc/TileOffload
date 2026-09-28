module tileoff_axpby_f64_kernel
contains

  subroutine axpby_f64_prepare(a, b, c)
    real(8) :: a(:), b(:), c(:)

    !$tileoff update device(a)
    !$tileoff update device(b)
    !$tileoff update device(c)
  end subroutine

  subroutine axpby_f64_compute(alpha, beta, a, b, c)
    real(8) :: alpha, beta
    real(8) :: a(:), b(:), c(:)
    integer :: i

    !$tileoff parallel tile(128) pack(a:device, b:device, c:device)
    do i = 1, size(c)
      c(i) = alpha * a(i) + beta * b(i)
    end do
  end subroutine

  subroutine axpby_f64_fetch(c)
    real(8) :: c(:)

    !$tileoff update host(c)
  end subroutine

  subroutine axpby_f64_release(a, b, c)
    real(8) :: a(:), b(:), c(:)

    !$tileoff release(a, b, c)
  end subroutine

end module

