import { cacheSampleData } from '../src/lib/utils/data-loader.server';

cacheSampleData()
  .then(() => console.log('Sample data cached successfully'))
  .catch(console.error);