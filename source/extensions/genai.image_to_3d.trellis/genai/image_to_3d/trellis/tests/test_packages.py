import omni.kit.test
from genai.image_to_3d.trellis import check_packages

# Having a test class derived from omni.kit.test.AsyncTestCase declared on the root of the module
# will make it auto-discoverable by omni.kit.test
class TestPackages(omni.kit.test.AsyncTestCase):
    # Before running each test
    async def setUp(self):
        pass

    # After running each test
    async def tearDown(self):
        pass

    # Actual test, notice it is an "async" function, so "await" can be used if needed
    async def test_packages(self):
        r = check_packages()
        self.assertTrue(r)
