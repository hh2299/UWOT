# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.




# def print_hi(name):
#     # Use a breakpoint in the code line below to debug your script.
#     print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.
#
#
# # Press the green button in the gutter to run the script.
# if __name__ == '__main__':
#     print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
class Solution(object):
    def rotateS(self, nums):
        for i in range(int(len(nums)/2)):
            temp = nums[i]
            nums[i] = nums[-i-1]
            nums[-i-1] = temp
        return nums

    def rotate(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: None Do not return anything, modify nums in-place instead.
        """
        nums = self.rotateS(nums[:len(nums)-k]) + self.rotateS(nums[-k:])
        nums = self.rotateS(nums)
        return nums


if __name__ == '__main__':
    solu = Solution()
    print(solu.rotate([1,2,3,4,5,6,7], 3))