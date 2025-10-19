import { useState, useEffect } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card'
import { Button } from './ui/button'
import { Input } from './ui/input'
import { Label } from './ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select'
import { Switch } from './ui/switch'
import { AlertThreshold, Device } from '../types/digital-twin'
import { blink } from '../blink/client'
import { Plus, Settings, Trash2 } from 'lucide-react'
import { useToast } from '../hooks/use-toast'

interface ThresholdSettingsProps {
  userId: string
  onClose: () => void
}

export function ThresholdSettings({ userId, onClose }: ThresholdSettingsProps) {
  const [thresholds, setThresholds] = useState<AlertThreshold[]>([])
  const [loading, setLoading] = useState(true)
  const { toast } = useToast()

  useEffect(() => {
    loadThresholds()
  }, [])

  const loadThresholds = async () => {
    try {
      const data = await (blink.db as any).alertThresholds.list({
        where: { userId },
        orderBy: { createdAt: 'desc' }
      })
      setThresholds(data)
    } catch (error) {
      console.error('Error loading thresholds:', error)
      toast({
        title: "Error",
        description: "Failed to load thresholds",
        variant: "destructive"
      })
    } finally {
      setLoading(false)
    }
  }

  const createThreshold = async () => {
    const newThreshold: AlertThreshold = {
      id: `THRESH_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      deviceType: 'temperature_sensor',
      metricType: 'temperature',
      operator: 'gt',
      value: 30,
      severity: 'warning',
      enabled: true,
      userId
    }

    try {
      await (blink.db as any).alertThresholds.create(newThreshold)
      setThresholds([newThreshold, ...thresholds])
      toast({
        title: "Success",
        description: "New threshold created"
      })
    } catch (error) {
      console.error('Error creating threshold:', error)
      toast({
        title: "Error",
        description: "Failed to create threshold",
        variant: "destructive"
      })
    }
  }

  const updateThreshold = async (id: string, updates: Partial<AlertThreshold>) => {
    try {
      await (blink.db as any).alertThresholds.update(id, updates)
      setThresholds(thresholds.map(t => 
        t.id === id ? { ...t, ...updates } : t
      ))
    } catch (error) {
      console.error('Error updating threshold:', error)
      toast({
        title: "Error",
        description: "Failed to update threshold",
        variant: "destructive"
      })
    }
  }

  const deleteThreshold = async (id: string) => {
    try {
      await (blink.db as any).alertThresholds.delete(id)
      setThresholds(thresholds.filter(t => t.id !== id))
      toast({
        title: "Success",
        description: "Threshold deleted"
      })
    } catch (error) {
      console.error('Error deleting threshold:', error)
      toast({
        title: "Error",
        description: "Failed to delete threshold",
        variant: "destructive"
      })
    }
  }

  const deviceTypes: Device['type'][] = [
    'temperature_sensor',
    'pressure_sensor', 
    'vibration_sensor',
    'humidity_sensor',
    'power_meter'
  ]

  const operators = [
    { value: 'gt', label: 'Greater than (>)' },
    { value: 'gte', label: 'Greater than or equal (≥)' },
    { value: 'lt', label: 'Less than (<)' },
    { value: 'lte', label: 'Less than or equal (≤)' },
    { value: 'eq', label: 'Equal to (=)' }
  ]

  const severities = [
    { value: 'info', label: 'Info' },
    { value: 'warning', label: 'Warning' },
    { value: 'critical', label: 'Critical' }
  ]

  if (loading) {
    return (
      <Card className="w-full max-w-4xl">
        <CardContent className="p-8">
          <div className="flex items-center justify-center">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
          </div>
        </CardContent>
      </Card>
    )
  }

  return (
    <Card className="w-full max-w-4xl">
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="flex items-center gap-2">
              <Settings className="h-5 w-5" />
              Alert Thresholds
            </CardTitle>
            <CardDescription>
              Configure alert thresholds for different device types
            </CardDescription>
          </div>
          <div className="flex gap-2">
            <Button onClick={createThreshold} className="flex items-center gap-2">
              <Plus className="h-4 w-4" />
              Add Threshold
            </Button>
            <Button variant="outline" onClick={onClose}>
              Close
            </Button>
          </div>
        </div>
      </CardHeader>

      <CardContent className="space-y-4">
        {thresholds.length === 0 ? (
          <div className="text-center py-8 text-muted-foreground">
            <p>No thresholds configured</p>
            <Button onClick={createThreshold} className="mt-4">
              Create your first threshold
            </Button>
          </div>
        ) : (
          thresholds.map(threshold => (
            <div key={threshold.id} className="border rounded-lg p-4 space-y-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-4">
                  <Switch
                    checked={Number(threshold.enabled) > 0}
                    onCheckedChange={(enabled) => 
                      updateThreshold(threshold.id, { enabled })
                    }
                  />
                  <Label className="text-sm font-medium">
                    {Number(threshold.enabled) > 0 ? 'Enabled' : 'Disabled'}
                  </Label>
                </div>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => deleteThreshold(threshold.id)}
                  className="text-red-600 hover:text-red-700"
                >
                  <Trash2 className="h-4 w-4" />
                </Button>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                <div>
                  <Label htmlFor={`device-type-${threshold.id}`}>Device Type</Label>
                  <Select
                    value={threshold.deviceType}
                    onValueChange={(value: Device['type']) =>
                      updateThreshold(threshold.id, { deviceType: value })
                    }
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {deviceTypes.map(type => (
                        <SelectItem key={type} value={type}>
                          {type.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>

                <div>
                  <Label htmlFor={`metric-${threshold.id}`}>Metric</Label>
                  <Input
                    id={`metric-${threshold.id}`}
                    value={threshold.metricType}
                    onChange={(e) =>
                      updateThreshold(threshold.id, { metricType: e.target.value })
                    }
                    placeholder="e.g. temperature"
                  />
                </div>

                <div>
                  <Label htmlFor={`operator-${threshold.id}`}>Condition</Label>
                  <Select
                    value={threshold.operator}
                    onValueChange={(value: AlertThreshold['operator']) =>
                      updateThreshold(threshold.id, { operator: value })
                    }
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {operators.map(op => (
                        <SelectItem key={op.value} value={op.value}>
                          {op.label}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>

                <div>
                  <Label htmlFor={`value-${threshold.id}`}>Threshold Value</Label>
                  <Input
                    id={`value-${threshold.id}`}
                    type="number"
                    value={threshold.value}
                    onChange={(e) =>
                      updateThreshold(threshold.id, { value: parseFloat(e.target.value) })
                    }
                  />
                </div>

                <div>
                  <Label htmlFor={`severity-${threshold.id}`}>Severity</Label>
                  <Select
                    value={threshold.severity}
                    onValueChange={(value: AlertThreshold['severity']) =>
                      updateThreshold(threshold.id, { severity: value })
                    }
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {severities.map(sev => (
                        <SelectItem key={sev.value} value={sev.value}>
                          <span className={`inline-block w-2 h-2 rounded-full mr-2 ${
                            sev.value === 'critical' ? 'bg-red-500' :
                            sev.value === 'warning' ? 'bg-yellow-500' :
                            'bg-blue-500'
                          }`}></span>
                          {sev.label}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
              </div>
            </div>
          ))
        )}
      </CardContent>
    </Card>
  )
}