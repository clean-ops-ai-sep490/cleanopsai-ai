import http from 'k6/http';
import { sleep } from 'k6';

export const options = {
  vus: 200,          // số user ảo
  duration: '30s',  // thời gian test
};

export default function () {
  http.get('https://heartlink-vercel.vercel.app/');
  sleep(1);
}