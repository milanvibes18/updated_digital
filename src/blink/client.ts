import { createClient } from '@blinkdotnew/sdk'

export const blink = createClient({
  projectId: 'digital-twin-iiot-platform-6r64p77c',
  authRequired: false
})

export default blink