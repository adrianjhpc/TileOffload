module tileoff_reduction_sum_kernel
contains

  subroutine reduction_sum_prepare(a)
    real :: a(:)

    !$tileoff enter data copyin(a)
  end subroutine

  subroutine reduction_sum_compute(a, result)
    real :: a(:)
    real :: result
    integer :: i

    result = 0.0

    !$tileoff parallel tile(256) reduction(+:result)
    do i = 1, size(a)
      result = result + a(i)
    end do
  end subroutine

  subroutine reduction_sum_release(a)
    real :: a(:)

    !$tileoff exit data delete(a)
  end subroutine

end module

